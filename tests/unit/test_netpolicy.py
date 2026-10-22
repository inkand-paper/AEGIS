"""The host allowlist (design doc 5.4)."""

import pytest

from aegis.netpolicy import HostNotAllowed, check_url, normalise_hosts


def test_hosts_are_cleaned_sorted_and_deduplicated() -> None:
    assert normalise_hosts(["Example.COM", "example.com", " b.org "]) == ["b.org", "example.com"]


@pytest.mark.parametrize(
    "bad",
    ["", "*", "*.", "http://example.com", "example.com/path", "user@example.com", "a b",
     "*.com.*", "exam*ple.com", "example.com:0", "example.com:99999", "example.com:x",
     "*.localhost", "*.127.0.0.1", "::1"],
)  # fmt: skip
def test_bad_patterns_are_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        normalise_hosts([bad])


def test_an_exact_host_matches_only_itself() -> None:
    allowed = {"example.com"}
    assert check_url("https://example.com/a?b=1", allowed) == "example.com"
    assert check_url("http://EXAMPLE.com./", allowed) == "example.com"
    for url in ("https://www.example.com/", "https://example.com.evil.org/", "https://evil.org/"):
        with pytest.raises(HostNotAllowed):
            check_url(url, allowed)


def test_a_wildcard_covers_subdomains_but_not_the_bare_domain() -> None:
    allowed = {"*.example.com"}
    assert check_url("https://a.b.example.com/", allowed) == "a.b.example.com"
    with pytest.raises(HostNotAllowed):
        check_url("https://example.com/", allowed)
    with pytest.raises(HostNotAllowed):
        check_url("https://notexample.com/", allowed)
    with pytest.raises(HostNotAllowed):
        check_url("https://evilexample.com/", allowed)


def test_a_port_restricts_the_match() -> None:
    allowed = {"localhost:8000"}
    assert check_url("http://localhost:8000/x", allowed) == "localhost"
    for url in ("http://localhost/", "http://localhost:8001/", "https://localhost:8000/"):
        if url == "https://localhost:8000/":
            assert check_url(url, allowed) == "localhost"  # same port number, scheme is free
            continue
        with pytest.raises(HostNotAllowed):
            check_url(url, allowed)


def test_default_ports_are_understood() -> None:
    assert check_url("https://example.com/", {"example.com:443"}) == "example.com"
    assert check_url("http://example.com/", {"example.com:80"}) == "example.com"
    with pytest.raises(HostNotAllowed):
        check_url("http://example.com/", {"example.com:443"})


@pytest.mark.parametrize(
    "url",
    ["http://localhost/", "http://127.0.0.1/", "http://10.0.0.5/", "http://192.168.1.1/",
     "http://169.254.169.254/latest/meta-data", "http://[::1]/", "http://printer.local/"],
)  # fmt: skip
def test_local_and_private_addresses_are_refused_even_with_a_wildcard(url: str) -> None:
    with pytest.raises(HostNotAllowed, match="local or private"):
        check_url(url, {"*.example.com", "*.local", "example.com"})


def test_a_private_address_works_only_when_named_exactly() -> None:
    assert check_url("http://127.0.0.1:9000/", {"127.0.0.1:9000"}) == "127.0.0.1"
    assert check_url("http://127.0.0.1/", {"127.0.0.1"}) == "127.0.0.1"


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "javascript:alert(1)", "data:text/html,hi", "ftp://example.com/",
     "//example.com/", "example.com", "https://user:pw@example.com/", "https://:80/"],
)  # fmt: skip
def test_unusual_urls_are_refused(url: str) -> None:
    with pytest.raises(HostNotAllowed):
        check_url(url, {"example.com"})


def test_no_allowed_hosts_means_no_network() -> None:
    with pytest.raises(HostNotAllowed, match="no allowed hosts"):
        check_url("https://example.com/", set())
