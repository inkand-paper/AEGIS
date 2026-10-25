"""Turn a pytest JUnit file into GitHub annotations, so a failing CI test names itself on the run
page (and in the API) without opening the log. Usage: python scripts/ci_annotate.py junit.xml"""

import sys
import xml.etree.ElementTree as ET  # noqa: S405 - our own pytest output, not untrusted input


def escape(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(path: str) -> int:
    try:
        root = ET.parse(path).getroot()  # noqa: S314
    except (OSError, ET.ParseError) as exc:
        print(f"::warning::no readable test report ({exc})")
        return 0
    failed = 0
    for case in root.iter("testcase"):
        for kind in ("failure", "error"):
            node = case.find(kind)
            if node is None:
                continue
            failed += 1
            name = f"{case.get('classname', '')}::{case.get('name', '')}"
            body = (node.text or "").strip()
            # The END of a traceback names the failing line; the start only repeats the test.
            detail = (node.get("message") or "")[:300] + "\n...\n" + body[-1100:]
            print(f"::error title={escape(name)[:200]}::{escape(detail.strip())}")
    print(f"{failed} failing test(s) annotated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
