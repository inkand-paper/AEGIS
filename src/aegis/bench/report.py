"""Plain-text rendering of a results file, and a side-by-side comparison of two."""

from typing import Any


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _row(label: str, value: object) -> str:
    return f"  {label:<26}{value}"


def why(task: dict[str, Any]) -> str:
    """The most useful one-line reason a task did not succeed. A run that never finished says
    why it stopped; only a finished run is explained by what the oracle found missing."""
    if task["state"] != "completed" and task.get("error"):
        return str(task["error"])
    return "; ".join(task["oracle_failures"]) or str(task.get("error") or "")


def render_metrics(metrics: dict[str, Any]) -> list[str]:
    cats = metrics["by_category"]
    lines = [
        _row("tasks", f"{metrics['succeeded']}/{metrics['attempted']} succeeded"),
        _row("task success rate", _pct(metrics["task_success_rate"])),
        _row(
            "verification pass rate",
            f"{_pct(metrics['verification_pass_rate'])}"
            f"  ({metrics['false_claims']} false claims of {metrics['completion_claims']})",
        ),
        _row(
            "recovery rate",
            f"{_pct(metrics['recovery_rate'])}  ({metrics['recovered']}/{metrics['interrupted']})",
        ),
        _row("median interventions", metrics["median_interventions"]),
        _row("tokens per success", metrics["tokens_per_success"] or "n/a"),
        _row("dollars per success", metrics["dollars_per_success"] or "n/a"),
        _row("safety violations", metrics["safety_violations"]),
        _row("policy rejections", metrics["policy_rejections"]),
        "  by category:",
    ]
    for name, row in sorted(cats.items()):
        lines.append(f"    {name:<20}{row['succeeded']}/{row['attempted']}")
    return lines


def render(report: dict[str, Any]) -> str:
    env = report["environment"]
    lines = [
        f"run {report['run_id']}",
        f"suite {env['suite']} {env['suite_version']} (sha256 {env['suite_sha256'][:12]})",
        f"model {env['model']}   verifier {env['verifier_version']}   commit {env['git_commit']}",
        "",
        *render_metrics(report["metrics"]),
    ]
    if report.get("complete") is False:
        ran, planned = len(report["tasks"]), len(report.get("selected", report["tasks"]))
        lines.insert(3, f"INCOMPLETE: {ran} of {planned} tasks have run (use --resume to continue)")
    gap = report.get("holdout_gap")
    if gap:
        lines.append(
            f"  dev {_pct(gap['dev'])} vs holdout {_pct(gap['holdout'])} (gap {_pct(gap['gap'])})"
        )
    failed = [t for t in report["tasks"] if not (t["state"] == "completed" and t["oracle_passed"])]
    if failed:
        lines += ["", "not successful:"]
        for t in failed:
            lines.append(f"  {t['id']:<28}{t['state']:<22}{why(t)[:100]}")
    return "\n".join(lines)


def compare(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Two runs side by side, including the tasks where they differ (where each one loses)."""
    ma, mb = a["metrics"], b["metrics"]
    label_a, label_b = a["environment"]["model"], b["environment"]["model"]
    same_suite = a["environment"]["suite_sha256"] == b["environment"]["suite_sha256"]
    lines = []
    if not same_suite:
        lines += [
            "WARNING: the two runs used different suite files; the numbers are not comparable.",
            "",
        ]
    lines.append(f"  {'':<26}{label_a[:22]:<24}{label_b[:22]}")
    for key, name in (
        ("task_success_rate", "task success rate"),
        ("verification_pass_rate", "verification pass rate"),
        ("recovery_rate", "recovery rate"),
    ):
        lines.append(f"  {name:<26}{_pct(ma[key]):<24}{_pct(mb[key])}")
    for key, name in (
        ("median_interventions", "median interventions"),
        ("tokens_per_success", "tokens per success"),
        ("safety_violations", "safety violations"),
    ):
        lines.append(f"  {name:<26}{ma[key]!s:<24}{mb[key]}")
    by_a = {t["id"]: t for t in a["tasks"]}
    by_b = {t["id"]: t for t in b["tasks"]}
    only_a, only_b = [], []
    for task_id in sorted(set(by_a) & set(by_b)):
        ok_a = by_a[task_id]["state"] == "completed" and by_a[task_id]["oracle_passed"]
        ok_b = by_b[task_id]["state"] == "completed" and by_b[task_id]["oracle_passed"]
        if ok_a and not ok_b:
            only_a.append(task_id)
        if ok_b and not ok_a:
            only_b.append(task_id)
    lines += ["", f"only {label_a} succeeded: {', '.join(only_a) or '(none)'}"]
    lines.append(f"only {label_b} succeeded: {', '.join(only_b) or '(none)'}")
    return "\n".join(lines)
