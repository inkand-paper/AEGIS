"""Browser tools (design doc sections 5.4 and 6, Phase 5).

The browser is a real headless Chromium driven through Playwright. Three tools only:

    browser_navigate   open a URL on an allowed host
    browser_read       read the current page again (text and the list of clickable elements)
    browser_click      click one listed element, and say what the page must show afterwards

What keeps this contained:
  * Every request the page makes, not only the first one, goes through the host allowlist
    (`aegis.netpolicy`). Redirects (checked hop by hop), images, scripts, frames and XHR to any
    other host are aborted, and the aborted hosts are reported back to the agent.
  * A click must state its post-condition (text the page must show, or part of the URL). If the
    page does not show it, the call FAILS with what the page shows instead. The agent is not
    allowed to assume a click worked. A failed click is not repeated: the controller refuses an
    identical call and the agent has to change course (or the task is replanned).
  * There is deliberately no tool that types text. The agent cannot fill in a form, log in or
    enter a credential. It can only read and follow links or press buttons.
  * Downloads, new windows, service workers, WebSockets and browser permissions are blocked.

The browser lives for one run of a task. After an approval pause or a resume the page is gone and
the agent has to navigate again. That is deliberate: a half-remembered page is how wrong clicks
happen.

Allowing a host lets the agent click ANYTHING on that host's pages that a visitor could click.
Only allow hosts where that is acceptable.
"""

import asyncio
import contextlib
import logging
import os
import time
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aegis.netpolicy import HostNotAllowed
from aegis.netpolicy import check_url as _check_url
from aegis.tools.base import PolicyError, Tool, ToolContext, ToolError, ToolUnavailable

logger = logging.getLogger(__name__)

# Use this Chromium instead of the one Playwright manages (CI images, unusual installs).
BROWSER_PATH_ENV = "AEGIS_BROWSER_EXECUTABLE"
INSTALL_HINT = "Chromium is not available. Install it once with: uv run playwright install chromium"

MAX_TEXT_CHARS = 2500
MAX_ELEMENTS = 40
NAVIGATE_TIMEOUT_MS = 15_000
CLICK_TIMEOUT_MS = 5_000
POST_CONDITION_WAIT_S = 5.0

_LIST_ELEMENTS_JS = """
(max) => {
  document.querySelectorAll('[data-aegis-id]').forEach(e => e.removeAttribute('data-aegis-id'));
  const visible = e => {
    const r = e.getBoundingClientRect();
    const s = getComputedStyle(e);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
  };
  const out = [];
  const selector = 'a[href], button, [role=button], input[type=submit], input[type=button]';
  for (const e of document.querySelectorAll(selector)) {
    if (out.length >= max) break;
    if (!visible(e) || e.disabled) continue;
    const text = (e.innerText || e.value || e.getAttribute('aria-label') || '')
      .trim().replace(/\\s+/g, ' ').slice(0, 80);
    if (!text) continue;
    const id = out.length;
    e.setAttribute('data-aegis-id', String(id));
    out.push({id, kind: e.tagName === 'A' ? 'link' : 'button', text,
              href: e.tagName === 'A' ? e.href : null});
  }
  return out;
}
"""

_ELEMENT_TEXT_JS = """
(e) => (e.innerText || e.value || e.getAttribute('aria-label') || '')
  .trim().replace(/\\s+/g, ' ').slice(0, 80)
"""


def check_url(url: str, allowed: frozenset[str]) -> str:
    """The network policy, as the policy error the dispatcher records."""
    try:
        return _check_url(url, allowed)
    except HostNotAllowed as exc:
        raise PolicyError(str(exc)) from None


class Element(BaseModel):
    id: int
    kind: Literal["link", "button"]
    text: str
    href: str | None = None


class PageState(BaseModel):
    """What the page shows right now. This is the BROWSER_STATE evidence."""

    url: str
    title: str
    elements: list[Element]
    blocked_hosts: list[str]  # hosts the page asked for that the allowlist refused
    note: str | None = None
    truncated: bool = False
    text: str  # last on purpose: when a result is cut for the model, the text goes first


class _Session:
    def __init__(self, allowed: frozenset[str]) -> None:
        self.allowed = allowed
        self.playwright: Any = None
        self.browser: Any = None
        self.context: Any = None
        self.page: Any = None
        self.blocked: set[str] = set()
        self.popups_closed = 0


class BrowserManager:
    """Owns one browser per task. Shared by the three tools."""

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}

    def existing(self, task_id: str) -> _Session | None:
        return self._sessions.get(task_id)

    async def session(self, ctx: ToolContext) -> _Session:
        found = self._sessions.get(ctx.task_id)
        if found is not None:
            return found
        try:
            from playwright.async_api import async_playwright
        except ImportError:  # pragma: no cover - a declared dependency
            raise ToolUnavailable("the playwright package is not installed") from None
        session = _Session(ctx.allowed_hosts)
        try:
            session.playwright = await async_playwright().start()
            executable = os.environ.get(BROWSER_PATH_ENV) or None
            session.browser = await session.playwright.chromium.launch(
                headless=True, executable_path=executable
            )
        except Exception as exc:
            await self._dispose(session)
            if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc):
                raise ToolUnavailable(INSTALL_HINT) from None
            logger.warning("browser launch failed: %s", type(exc).__name__)
            raise ToolUnavailable(f"the browser could not start ({type(exc).__name__})") from None
        session.context = await session.browser.new_context(
            accept_downloads=False,
            service_workers="block",
            permissions=[],
            viewport={"width": 1280, "height": 800},
        )

        def refuse(url: str) -> bool:
            try:
                _check_url(url, session.allowed)
            except HostNotAllowed:
                session.blocked.add(urlsplit(url).hostname or url[:60])
                return True
            return False

        async def guard(route: Any) -> None:
            # `route.continue_()` would let Chromium follow a redirect to ANY host by itself, so
            # each request is fetched here without following redirects, the Location of a
            # redirect is checked, and only then is the answer handed to the page. The page then
            # requests the redirect target, which comes through this guard again.
            request = route.request
            if refuse(request.url):
                await route.abort("blockedbyclient")
                return
            try:
                response = await route.fetch(max_redirects=0)
            except Exception:
                await route.abort("failed")
                return
            location = response.headers.get("location")
            if 300 <= response.status < 400 and location:
                if refuse(urljoin(request.url, location)):
                    await route.abort("blockedbyclient")
                    return
            await route.fulfill(response=response)

        await session.context.route("**/*", guard)
        await session.context.route_web_socket("**/*", lambda socket: socket.close())

        def on_page(page: Any) -> None:
            if session.page is not None and page is not session.page:
                session.popups_closed += 1
                asyncio.ensure_future(page.close())

        session.context.on("page", on_page)
        session.page = await session.context.new_page()
        self._sessions[ctx.task_id] = session
        return session

    async def close_task(self, task_id: str) -> None:
        session = self._sessions.pop(task_id, None)
        if session is not None:
            await self._dispose(session)

    @staticmethod
    async def _dispose(session: _Session) -> None:
        for closer in (
            getattr(session.context, "close", None),
            getattr(session.browser, "close", None),
            getattr(session.playwright, "stop", None),
        ):
            if closer is None:
                continue
            try:
                await closer()
            except Exception:  # best effort: a dead browser must not hide the real result
                logger.debug("browser cleanup step failed", exc_info=True)


async def _page_state(session: _Session, note: str | None = None) -> PageState:
    page = session.page
    try:
        text = await page.inner_text("body", timeout=3000)
    except Exception:
        text = ""
    text = " ".join(text.split())
    truncated = len(text) > MAX_TEXT_CHARS
    elements = await page.evaluate(_LIST_ELEMENTS_JS, MAX_ELEMENTS)
    if session.popups_closed:
        extra = f"{session.popups_closed} new window(s) were blocked"
        note = f"{note}; {extra}" if note else extra
        session.popups_closed = 0
    return PageState(
        url=page.url,
        title=await page.title(),
        elements=[Element.model_validate(e) for e in elements],
        blocked_hosts=sorted(session.blocked)[:10],
        note=note,
        truncated=truncated,
        text=text[:MAX_TEXT_CHARS],
    )


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NavigateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=2000, description="http(s) URL on an allowed host")


class ClickArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element: int = Field(ge=0, description="the id of the element, from the last page listing")
    text: str = Field(min_length=1, max_length=200, description="that element's text, exactly")
    expect_text: str | None = Field(
        default=None, min_length=1, max_length=200, description="text the page must show after"
    )
    expect_url_contains: str | None = Field(
        default=None, min_length=1, max_length=200, description="part of the URL after the click"
    )

    @model_validator(mode="after")
    def _needs_a_post_condition(self) -> "ClickArgs":
        if self.expect_text is None and self.expect_url_contains is None:
            raise ValueError(
                "say what the page must show after the click: give expect_text or "
                "expect_url_contains"
            )
        return self


def _no_page() -> ToolError:
    return ToolError("no page is open. Call browser_navigate with a URL first.")


class BrowserNavigateTool(Tool[NavigateArgs, PageState]):
    name = "browser_navigate"
    description = (
        "Open a web page in a browser and return its text and its clickable elements. Only hosts "
        "the task was allowed to use can be opened; anything else is refused. You cannot type "
        "into pages."
    )
    arg_model = NavigateArgs
    result_model = PageState
    side_effects = "external"
    requires_network = True
    idempotent = True  # loading a page twice is harmless
    timeout_s = 45.0

    def __init__(self, manager: BrowserManager) -> None:
        self._manager = manager

    async def execute(self, args: NavigateArgs, ctx: ToolContext) -> PageState:
        check_url(args.url, ctx.allowed_hosts)  # a clear PolicyError before any browser starts
        session = await self._manager.session(ctx)
        try:
            response = await session.page.goto(
                args.url, wait_until="load", timeout=NAVIGATE_TIMEOUT_MS
            )
        except Exception as exc:
            reason = str(exc).split("\n", 1)[0][:200]
            raise ToolError(f"could not open {args.url}: {reason}") from None
        status = response.status if response is not None else None
        if status is not None and status >= 400:
            raise ToolError(
                f"{args.url} answered HTTP {status}. The page does not exist or is not available."
            )
        return await _page_state(session, note=f"HTTP {status}" if status else None)

    async def close_task(self, task_id: str) -> None:
        await self._manager.close_task(task_id)


class BrowserReadTool(Tool[_NoArgs, PageState]):
    name = "browser_read"
    description = (
        "Read the page that is open right now: its text and its clickable elements with ids. "
        "Use it after a failed click to see what the page really shows."
    )
    arg_model = _NoArgs
    result_model = PageState
    side_effects = "none"
    requires_network = True  # only offered to tasks that were allowed a host
    idempotent = True
    timeout_s = 30.0

    def __init__(self, manager: BrowserManager) -> None:
        self._manager = manager

    async def execute(self, args: _NoArgs, ctx: ToolContext) -> PageState:
        session = self._manager.existing(ctx.task_id)
        if session is None or session.page is None:
            raise _no_page()
        return await _page_state(session)


class BrowserClickTool(Tool[ClickArgs, PageState]):
    name = "browser_click"
    description = (
        "Click one element from the page listing (by its id, plus its exact text as a check). "
        "You MUST say what the page should show afterwards (expect_text or expect_url_contains). "
        "If it does not, the call fails and tells you what the page shows instead. A click that "
        "failed will not be run again: choose something different."
    )
    arg_model = ClickArgs
    result_model = PageState
    side_effects = "external"
    requires_network = True
    idempotent = False  # a click may submit, buy or delete
    repeatable_after_failure = False
    timeout_s = 45.0

    def __init__(self, manager: BrowserManager) -> None:
        self._manager = manager

    async def execute(self, args: ClickArgs, ctx: ToolContext) -> PageState:
        session = self._manager.existing(ctx.task_id)
        if session is None or session.page is None:
            raise _no_page()
        page = session.page
        locator = page.locator(f'[data-aegis-id="{args.element}"]')
        if await locator.count() != 1:
            raise ToolError(
                f"element {args.element} is not on the page any more. "
                "Call browser_read to get the current elements."
            )
        actual = " ".join((await locator.evaluate(_ELEMENT_TEXT_JS)).split())
        if actual != " ".join(args.text.split()):
            raise ToolError(
                f"element {args.element} is {actual!r}, not {args.text!r}. "
                "The page changed. Call browser_read to get the current elements."
            )
        href = await locator.get_attribute("href")
        if href:
            check_url(await locator.evaluate("e => e.href"), ctx.allowed_hosts)
        try:
            await locator.click(timeout=CLICK_TIMEOUT_MS)
        except Exception as exc:
            reason = str(exc).split("\n", 1)[0][:200]
            raise ToolError(f"could not click {args.text!r}: {reason}") from None
        with contextlib.suppress(Exception):  # a click that stays on the page is normal
            await page.wait_for_load_state("load", timeout=CLICK_TIMEOUT_MS)
        problem = await self._wait_for_post_condition(session, args)
        if problem is not None:
            state = await _page_state(session)
            raise ToolError(
                f"clicked {args.text!r}, but the page does not show what you expected "
                f"({problem}). The page is now {state.url!r} titled {state.title!r}. "
                "Call browser_read to see it, then do something different or report that this "
                "cannot be done."
            )
        return await _page_state(session, note="the expected result is on the page")

    @staticmethod
    async def _wait_for_post_condition(session: _Session, args: ClickArgs) -> str | None:
        deadline = time.monotonic() + POST_CONDITION_WAIT_S
        page = session.page
        missing: list[str] = []
        while True:
            missing = []
            if args.expect_url_contains is not None and args.expect_url_contains not in page.url:
                missing.append(f"URL does not contain {args.expect_url_contains!r}")
            if args.expect_text is not None:
                try:
                    body = await page.inner_text("body", timeout=1000)
                except Exception:
                    body = ""
                if args.expect_text.lower() not in " ".join(body.split()).lower():
                    missing.append(f"text {args.expect_text!r} is not shown")
            if not missing:
                return None
            if time.monotonic() >= deadline:
                return "; ".join(missing)
            await asyncio.sleep(0.1)


def browser_tools() -> list[Tool[Any, Any]]:
    manager = BrowserManager()
    return [
        BrowserNavigateTool(manager),
        BrowserReadTool(manager),
        BrowserClickTool(manager),
    ]
