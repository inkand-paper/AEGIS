"""Phase 5 browser tools against a real headless Chromium and local test sites.

These tests need Chromium (`uv run playwright install chromium`, or AEGIS_BROWSER_EXECUTABLE).
Without it they are skipped, and the one test of the "not installed" path still runs.
"""

import asyncio
import os
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from fakes import ScriptedModel, call_tools, plan_json, say

from aegis.controller import Controller
from aegis.executor import Executor
from aegis.memory import EventLog
from aegis.models import EventType, ToolCall, ToolOutcome
from aegis.tools import default_registry
from aegis.tools.base import PolicyError, ToolContext, ToolError, ToolUnavailable
from aegis.tools.browser import (
    BROWSER_PATH_ENV,
    BrowserClickTool,
    BrowserManager,
    BrowserNavigateTool,
    BrowserReadTool,
    ClickArgs,
    NavigateArgs,
    PageState,
    _NoArgs,
)

# ------------------------------------------------------------------ the test sites


class Site:
    """Two local servers on different ports: `main` is allowed in tests, `other` never is."""

    def __init__(self) -> None:
        self.hits: dict[str, int] = {}
        self.other_hits: dict[str, int] = {}
        self._servers: list[ThreadingHTTPServer] = []
        self.other = self._start(self._other_handler())
        self.main = self._start(self._main_handler())

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.main.server_port}"

    @property
    def other_origin(self) -> str:
        return f"http://127.0.0.1:{self.other.server_port}"

    @property
    def allow(self) -> list[str]:
        return [f"127.0.0.1:{self.main.server_port}"]

    def _start(self, handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self._servers.append(server)
        return server

    def stop(self) -> None:
        for server in self._servers:
            server.shutdown()
            server.server_close()

    def _other_handler(self) -> type[BaseHTTPRequestHandler]:
        site = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                site.other_hits[self.path] = site.other_hits.get(self.path, 0) + 1
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><title>Other</title><body>The other site</body></html>")

            def log_message(self, *args: Any) -> None:
                pass

        return Handler

    def _main_handler(self) -> type[BaseHTTPRequestHandler]:
        site = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.split("?")[0]
                site.hits[path] = site.hits.get(path, 0) + 1
                if path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", f"{site.other_origin}/landed")
                    self.end_headers()
                    return
                page = PAGES.get(path)
                if page is None:
                    self.send_response(404)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"<html><title>Missing</title><body>Not here</body></html>")
                    return
                body = page.replace("{OTHER}", site.other_origin).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        return Handler


PAGES = {
    "/": """<html><head><title>Shop</title></head><body>
        <h1>Welcome to the shop</h1>
        <a href="/products">Products</a>
        <button onclick="document.getElementById('msg').textContent='Clicked once'">
          Say hi
        </button>
        <a href="{OTHER}/away">Leave the site</a>
        <a href="/order">Order now</a>
        <a href="/popup">Popups</a>
        <p id="msg"></p>
        <img src="{OTHER}/pixel.png">
        </body></html>""",
    "/products": "<html><title>Products</title><body>Product list: Item 1, Item 2</body></html>",
    "/order": "<html><title>Order</title><body>Order placed. Thank you.</body></html>",
    "/popup": (
        '<html><title>Popup</title><body><a target="_blank" href="/products">Open</a></body></html>'
    ),
}


@pytest.fixture(scope="module")
def site() -> Iterator[Site]:
    found = Site()
    yield found
    found.stop()


def _chromium_available() -> bool:
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=os.environ.get(BROWSER_PATH_ENV) or None)
            browser.close()
    except Exception:
        return False
    return True


needs_chromium = pytest.mark.skipif(
    not _chromium_available(),
    reason="Chromium is not installed (uv run playwright install chromium)",
)

# ------------------------------------------------------------------ helpers


def tools() -> tuple[BrowserManager, BrowserNavigateTool, BrowserReadTool, BrowserClickTool]:
    manager = BrowserManager()
    return (
        manager,
        BrowserNavigateTool(manager),
        BrowserReadTool(manager),
        BrowserClickTool(manager),
    )


def ctx(tmp_path: Path, site: Site, hosts: list[str] | None = None) -> ToolContext:
    return ToolContext(
        task_id="t1",
        workspace=tmp_path,
        allowed_effects=frozenset({"external"}),
        allowed_hosts=frozenset(site.allow if hosts is None else hosts),
    )


def scenario(body: Callable[..., Any], tmp_path: Path, site: Site, **kwargs: Any) -> Any:
    async def go() -> Any:
        manager, nav, read, click = tools()
        context = ctx(tmp_path, site, **kwargs)
        try:
            return await body(nav, read, click, context)
        finally:
            await manager.close_task("t1")

    return asyncio.run(go())


def element(state: PageState, text: str) -> int:
    return next(e.id for e in state.elements if e.text == text)


# ------------------------------------------------------------------ reading and clicking


@needs_chromium
def test_navigate_returns_text_and_clickable_elements(tmp_path: Path, site: Site) -> None:
    async def body(nav: Any, read: Any, click: Any, c: ToolContext) -> PageState:
        return await nav.execute(NavigateArgs(url=site.origin + "/"), c)

    state = scenario(body, tmp_path, site)
    assert state.title == "Shop"
    assert "Welcome to the shop" in state.text
    assert [(e.kind, e.text) for e in state.elements] == [
        ("link", "Products"),
        ("button", "Say hi"),
        ("link", "Leave the site"),
        ("link", "Order now"),
        ("link", "Popups"),
    ]
    assert [e.id for e in state.elements] == [0, 1, 2, 3, 4]


@needs_chromium
def test_a_click_with_a_met_post_condition_returns_the_new_page(tmp_path: Path, site: Site) -> None:
    async def body(nav: Any, read: Any, click: Any, c: ToolContext) -> tuple[PageState, PageState]:
        home = await nav.execute(NavigateArgs(url=site.origin + "/"), c)
        args = ClickArgs(
            element=element(home, "Products"), text="Products", expect_url_contains="/products"
        )
        after = await click.execute(args, c)
        return after, await read.execute(_NoArgs(), c)

    after, reread = scenario(body, tmp_path, site)
    assert after.url.endswith("/products")
    assert "Product list" in after.text
    assert reread.url == after.url


@needs_chromium
def test_a_click_that_changes_the_page_without_navigating_is_seen(
    tmp_path: Path, site: Site
) -> None:
    async def body(nav: Any, read: Any, click: Any, c: ToolContext) -> PageState:
        home = await nav.execute(NavigateArgs(url=site.origin + "/"), c)
        args = ClickArgs(element=element(home, "Say hi"), text="Say hi", expect_text="clicked once")
        return await click.execute(args, c)

    assert "Clicked once" in scenario(body, tmp_path, site).text


@needs_chromium
def test_a_click_whose_post_condition_is_not_met_fails_and_says_what_is_shown(
    tmp_path: Path, site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("aegis.tools.browser.POST_CONDITION_WAIT_S", 0.3)

    async def body(nav: Any, read: Any, click: Any, c: ToolContext) -> str:
        home = await nav.execute(NavigateArgs(url=site.origin + "/"), c)
        args = ClickArgs(
            element=element(home, "Products"), text="Products", expect_text="Order placed"
        )
        with pytest.raises(ToolError) as info:
            await click.execute(args, c)
        return str(info.value)

    message = scenario(body, tmp_path, site)
    assert "'Order placed' is not shown" in message
    assert "/products" in message and "Products" in message  # what the page shows instead


@needs_chromium
def test_a_stale_or_wrong_element_is_refused_before_clicking(tmp_path: Path, site: Site) -> None:
    before = dict(site.hits)

    async def body(nav: Any, read: Any, click: Any, c: ToolContext) -> list[str]:
        await nav.execute(NavigateArgs(url=site.origin + "/"), c)
        out = []
        for args in (
            ClickArgs(element=0, text="Order now", expect_text="x"),  # id 0 is "Products"
            ClickArgs(element=99, text="Products", expect_text="x"),
        ):
            with pytest.raises(ToolError) as info:
                await click.execute(args, c)
            out.append(str(info.value))
        return out

    wrong_text, missing = scenario(body, tmp_path, site)
    assert "'Products', not 'Order now'" in wrong_text
    assert "not on the page" in missing
    assert site.hits.get("/products", 0) == before.get("/products", 0)  # nothing was clicked


def test_a_click_must_state_what_it_expects() -> None:
    with pytest.raises(ValueError, match="expect_text or expect_url_contains"):
        ClickArgs(element=0, text="Products")


@needs_chromium
def test_read_and_click_need_a_page_first(tmp_path: Path, site: Site) -> None:
    async def body(nav: Any, read: Any, click: Any, c: ToolContext) -> list[str]:
        out = []
        with pytest.raises(ToolError) as one:
            await read.execute(_NoArgs(), c)
        out.append(str(one.value))
        with pytest.raises(ToolError) as two:
            await click.execute(ClickArgs(element=0, text="x", expect_text="y"), c)
        out.append(str(two.value))
        return out

    for message in scenario(body, tmp_path, site):
        assert "browser_navigate" in message


@needs_chromium
def test_a_missing_page_is_an_error_not_evidence(tmp_path: Path, site: Site) -> None:
    async def body(nav: Any, read: Any, click: Any, c: ToolContext) -> str:
        with pytest.raises(ToolError) as info:
            await nav.execute(NavigateArgs(url=site.origin + "/nope"), c)
        return str(info.value)

    assert "HTTP 404" in scenario(body, tmp_path, site)


# ------------------------------------------------------------------ the network policy


@needs_chromium
@pytest.mark.parametrize(
    "path",
    ["{OTHER}/away", "file:///etc/passwd", "javascript:alert(1)", "http://example.com/"],
)
def test_navigation_outside_the_allowlist_is_refused(tmp_path: Path, site: Site, path: str) -> None:
    async def body(nav: Any, read: Any, click: Any, c: ToolContext) -> None:
        with pytest.raises(PolicyError):
            await nav.execute(NavigateArgs(url=path.replace("{OTHER}", site.other_origin)), c)

    scenario(body, tmp_path, site)
    assert site.other_hits.get("/away", 0) == 0


@needs_chromium
def test_subresources_on_other_hosts_are_blocked_and_reported(tmp_path: Path, site: Site) -> None:
    site.other_hits.clear()

    async def body(nav: Any, read: Any, click: Any, c: ToolContext) -> PageState:
        return await nav.execute(NavigateArgs(url=site.origin + "/"), c)

    state = scenario(body, tmp_path, site)
    assert site.other_hits == {}  # the page's <img> never reached the other server
    assert state.blocked_hosts == ["127.0.0.1"]


@needs_chromium
def test_a_redirect_to_another_host_is_blocked(tmp_path: Path, site: Site) -> None:
    site.other_hits.clear()

    async def body(nav: Any, read: Any, click: Any, c: ToolContext) -> str:
        with pytest.raises(ToolError) as info:
            await nav.execute(NavigateArgs(url=site.origin + "/redirect"), c)
        return str(info.value)

    assert "could not open" in scenario(body, tmp_path, site)
    assert site.other_hits == {}


@needs_chromium
def test_a_link_to_a_blocked_host_is_refused_before_the_click(tmp_path: Path, site: Site) -> None:
    site.other_hits.clear()

    async def body(nav: Any, read: Any, click: Any, c: ToolContext) -> None:
        home = await nav.execute(NavigateArgs(url=site.origin + "/"), c)
        args = ClickArgs(
            element=element(home, "Leave the site"), text="Leave the site", expect_text="Other"
        )
        with pytest.raises(PolicyError, match="explicitly allowed"):
            await click.execute(args, c)

    scenario(body, tmp_path, site)
    assert site.other_hits == {}


@needs_chromium
def test_new_windows_are_closed_and_reported(tmp_path: Path, site: Site) -> None:
    async def body(nav: Any, read: Any, click: Any, c: ToolContext) -> list[PageState]:
        await nav.execute(NavigateArgs(url=site.origin + "/popup"), c)
        state = await read.execute(_NoArgs(), c)
        args = ClickArgs(element=state.elements[0].id, text="Open", expect_text="Open")
        clicked = await click.execute(args, c)
        await asyncio.sleep(0.5)
        return [clicked, await read.execute(_NoArgs(), c)]

    states = scenario(body, tmp_path, site)
    assert all(s.url.endswith("/popup") for s in states)  # still on the original page
    # the window is reported by whichever answer comes after it was closed
    assert any("new window" in (s.note or "") for s in states)


@needs_chromium
def test_the_browser_is_released_when_the_task_ends(tmp_path: Path, site: Site) -> None:
    async def go() -> list[bool]:
        manager, nav, read, _ = tools()
        c = ctx(tmp_path, site)
        await nav.execute(NavigateArgs(url=site.origin + "/"), c)
        before = manager.existing("t1") is not None
        await manager.close_task("t1")
        return [before, manager.existing("t1") is None]

    assert asyncio.run(go()) == [True, True]


# ------------------------------------------------------------------ through the dispatcher


def test_without_allowed_hosts_the_browser_tools_are_refused_by_policy(tmp_path: Path) -> None:
    events = EventLog(tmp_path / "e.db")
    executor = Executor(default_registry(), events)
    c = ToolContext(task_id="t1", workspace=tmp_path, allowed_effects=frozenset({"external"}))
    call = ToolCall(
        id="c1",
        task_id="t1",
        step_id="s1",
        tool_name="browser_navigate",
        arguments={"url": "https://example.com/"},
        requested_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
    )
    result = asyncio.run(executor.execute(call, c))
    assert result.record.outcome is ToolOutcome.POLICY_VIOLATION
    assert "no host was allowed" in (result.record.error or "")


def test_a_missing_browser_is_reported_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(BROWSER_PATH_ENV, str(tmp_path / "no-such-chromium"))
    events = EventLog(tmp_path / "e.db")
    registry = default_registry()
    executor = Executor(registry, events)
    c = ToolContext(
        task_id="t1",
        workspace=tmp_path,
        allowed_effects=frozenset({"external"}),
        allowed_hosts=frozenset({"example.com"}),
    )
    from datetime import UTC, datetime

    call = ToolCall(
        id="c1",
        task_id="t1",
        step_id="s1",
        tool_name="browser_navigate",
        arguments={"url": "https://example.com/"},
        requested_at=datetime.now(UTC),
    )
    result = asyncio.run(executor.execute(call, c))
    asyncio.run(registry.close_task("t1"))
    assert result.record.outcome is ToolOutcome.UNAVAILABLE
    assert "playwright install chromium" in (result.record.error or "") or "could not start" in (
        result.record.error or ""
    )
    assert isinstance(ToolUnavailable("x"), ToolError)


# ------------------------------------------------------------------ the whole agent


@needs_chromium
def test_the_agent_replans_instead_of_repeating_a_failed_click(
    tmp_path: Path, site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 5 exit criterion: a click that cannot work is not retried; the plan changes."""
    monkeypatch.setattr("aegis.tools.browser.POST_CONDITION_WAIT_S", 0.3)
    site.hits.clear()
    events = EventLog(tmp_path / "aegis.db")
    wrong = ("browser_click", {"element": 0, "text": "Products", "expect_text": "Order placed"})
    right = ("browser_click", {"element": 3, "text": "Order now", "expect_text": "Order placed"})
    model = ScriptedModel(
        say(plan_json("Open the shop", "Press the checkout button")),
        call_tools(("browser_navigate", {"url": site.origin + "/"})),
        say("The shop is open."),
        call_tools(wrong),  # step 2, attempt 1: fails
        call_tools(wrong),  # the same call again: refused
        call_tools(wrong),  # attempt 2: refused
        call_tools(wrong),  # refused: the step is out of attempts
        say(plan_json("Open the shop again and press the link called Order now")),
        # the wrong click had already moved the browser to /products, so the shop is reopened
        call_tools(("browser_navigate", {"url": site.origin + "/"})),
        call_tools(right),
        say("The order page says: Order placed."),
    )
    controller = Controller(
        model,
        default_registry(),
        events,
        workspaces_dir=tmp_path / "workspaces",
        max_turns_per_step=2,
    )
    task = controller.create_task(
        "Order the item in the shop",
        acceptance=["page_contains:Order placed", "url_contains:/order"],
        allowed_hosts=site.allow,
    )
    finished = asyncio.run(controller.run(task.id))

    assert finished.state.value == "completed", finished.last_error
    outcomes = [e.payload["outcome"] for e in events.events(task.id, EventType.TOOL_RESULT)]
    assert outcomes == [
        "ok",
        "tool_error",
        "repeat_blocked",
        "repeat_blocked",
        "repeat_blocked",
        "ok",
        "ok",
    ]
    revised = events.events(task.id, EventType.PLAN_REVISED)
    assert [e.payload["trigger"] for e in revised] == ["step_failed"]
    assert site.hits.get("/products", 0) == 1  # the wrong click happened once, not four times
    assert site.hits.get("/order", 0) == 1
    kinds = {e.payload["kind"] for e in events.events(task.id, EventType.EVIDENCE_ADDED)}
    assert "browser_state" in kinds
    assert events.unmatched_tool_calls(task.id) == []
    assert controller._registry.get("browser_read")  # still registered
    assert all(
        not t._manager.existing(task.id) for t in (controller._registry.get("browser_navigate"),)
    )  # the browser was released when the run ended


@needs_chromium
def test_a_task_cannot_complete_on_a_page_the_browser_never_showed(
    tmp_path: Path, site: Site
) -> None:
    events = EventLog(tmp_path / "aegis.db")
    model = ScriptedModel(
        say(plan_json("Open the shop")),
        call_tools(("browser_navigate", {"url": site.origin + "/"})),
        say("Done. The order was placed."),  # a claim, not evidence
    )
    controller = Controller(
        model, default_registry(), events, workspaces_dir=tmp_path / "workspaces"
    )
    from aegis.models import Budget

    task = controller.create_task(
        "Order",
        acceptance=["page_contains:Order placed"],
        allowed_hosts=site.allow,
        budget=Budget(max_replans=0),
    )
    finished = asyncio.run(controller.run(task.id))
    assert finished.state.value == "failed"
    assert "VerificationFailed" in (finished.last_error or "")


# ------------------------------------------------------------------ more scripted workflows


def _agent(tmp_path: Path, model: ScriptedModel, **kwargs: Any) -> tuple[Controller, EventLog]:
    events = EventLog(tmp_path / "aegis.db")
    controller = Controller(
        model, default_registry(), events, workspaces_dir=tmp_path / "workspaces", **kwargs
    )
    return controller, events


@needs_chromium
def test_workflow_browse_and_click_through_to_a_page(tmp_path: Path, site: Site) -> None:
    model = ScriptedModel(
        say(plan_json("Open the shop and go to the product list")),
        call_tools(("browser_navigate", {"url": site.origin + "/"})),
        call_tools(
            (
                "browser_click",
                {"element": 0, "text": "Products", "expect_url_contains": "/products"},
            )
        ),
        say("The product list shows Item 1 and Item 2."),
    )
    controller, events = _agent(tmp_path, model)
    task = controller.create_task(
        "List the products",
        acceptance=["page_contains:Item 2", "url_contains:/products", "tool_used:browser_click"],
        allowed_hosts=site.allow,
    )
    finished = asyncio.run(controller.run(task.id))
    assert finished.state.value == "completed", finished.last_error


@needs_chromium
def test_workflow_a_button_that_only_changes_the_page(tmp_path: Path, site: Site) -> None:
    model = ScriptedModel(
        say(plan_json("Open the shop and press Say hi")),
        call_tools(("browser_navigate", {"url": site.origin + "/"})),
        call_tools(("browser_click", {"element": 1, "text": "Say hi", "expect_text": "Clicked"})),
        say("Pressed it."),
    )
    controller, events = _agent(tmp_path, model)
    task = controller.create_task(
        "Press the button", acceptance=["page_contains:Clicked once"], allowed_hosts=site.allow
    )
    finished = asyncio.run(controller.run(task.id))
    assert finished.state.value == "completed", finished.last_error


@needs_chromium
def test_workflow_an_attempt_to_leave_the_allowed_site_is_refused_and_recorded(
    tmp_path: Path, site: Site
) -> None:
    site.other_hits.clear()
    model = ScriptedModel(
        say(plan_json("Open the shop, then follow the link that leaves the site")),
        call_tools(("browser_navigate", {"url": site.origin + "/"})),
        call_tools(("browser_navigate", {"url": site.other_origin + "/away"})),  # injected idea
        say("I could not open the other site: it is not an allowed host."),
    )
    controller, events = _agent(tmp_path, model)
    task = controller.create_task(
        "Visit the shop", acceptance=["page_contains:Welcome"], allowed_hosts=site.allow
    )
    finished = asyncio.run(controller.run(task.id))
    assert finished.state.value == "completed", finished.last_error
    outcomes = [e.payload["outcome"] for e in events.events(task.id, EventType.TOOL_RESULT)]
    assert outcomes == ["ok", "policy_violation"]
    assert site.other_hits == {}
