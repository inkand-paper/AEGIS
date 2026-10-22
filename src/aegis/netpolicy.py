"""Network policy: which hosts a task may reach (design doc sections 3.2 and 5.4).

A task starts with NO network. `--allow-host example.com` lets it reach that host and nothing
else. A pattern is an exact host name, or `*.example.com` for the subdomains of example.com
(not example.com itself). Host names are compared case-insensitively. An optional `:port`
restricts the port.

Local and private addresses (localhost, 127.0.0.0/8, 10/8, 192.168/16, link-local and so on)
are refused unless that exact address is allowed. A wildcard never covers them.

Known limit: the check is on the name in the URL. A public name that is made to resolve to a
private address (DNS rebinding) is not caught here. The container sandbox of Phase 6 is what
closes that.
"""

import ipaddress
from collections.abc import Collection, Iterable
from urllib.parse import urlsplit

_SCHEMES = {"http": 80, "https": 443}


class HostNotAllowed(ValueError):
    """The URL is not one this task may reach. The message says why, for the model and the log."""


def _split(pattern: str) -> tuple[str, int | None]:
    text = pattern.strip().lower()
    if not text or "/" in text or "@" in text or " " in text:
        raise ValueError(f"not a host name: {pattern!r}")
    if text.count(":") == 1:
        host, _, port_text = text.partition(":")
        if not port_text.isdigit() or not 0 < int(port_text) < 65536:
            raise ValueError(f"bad port in {pattern!r}")
        return host, int(port_text)
    if ":" in text:  # a bare IPv6 address is not supported as a pattern
        raise ValueError(f"not a host name: {pattern!r}")
    return text, None


def normalise_hosts(patterns: Iterable[str]) -> list[str]:
    """Validate and clean the allowed hosts. Raises ValueError with a message for the user."""
    cleaned: list[str] = []
    for pattern in patterns:
        host, port = _split(str(pattern))
        if "*" in host:
            if not host.startswith("*.") or "*" in host[2:] or "." not in host[2:]:
                raise ValueError(f"wildcards must look like *.example.com: {pattern!r}")
            if _is_ip(host[2:]) or _is_private(host[2:]):
                raise ValueError(f"a wildcard cannot be used with a local address: {pattern!r}")
        entry = host if port is None else f"{host}:{port}"
        if entry not in cleaned:
            cleaned.append(entry)
    return sorted(cleaned)


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return True


def _is_private(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return not address.is_global


def check_url(url: str, allowed: Collection[str]) -> str:
    """Return the host of `url` if the task may reach it, else raise HostNotAllowed."""
    if not allowed:
        raise HostNotAllowed("this task has no allowed hosts")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise HostNotAllowed(f"not a valid URL: {url!r}") from None
    scheme = parts.scheme.lower()
    if scheme not in _SCHEMES:
        raise HostNotAllowed(f"only http and https are allowed, not {scheme or 'a relative URL'!r}")
    if parts.username or parts.password:
        raise HostNotAllowed("URLs with a user name or password are not allowed")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise HostNotAllowed(f"the URL has no host: {url!r}")
    effective_port = port if port is not None else _SCHEMES[scheme]
    private = _is_private(host)
    for entry in allowed:
        pattern, pattern_port = _split(entry)
        if pattern_port is not None and pattern_port != effective_port:
            continue
        if pattern.startswith("*."):
            if not private and host.endswith(pattern[1:]) and host != pattern[2:]:
                return host
        elif host == pattern:
            return host
    kind = "a local or private address that is not explicitly allowed" if private else "not allowed"
    raise HostNotAllowed(f"host {host!r} is {kind}. Allowed: {', '.join(sorted(allowed))}")
