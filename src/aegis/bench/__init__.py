"""Benchmark harness (design doc sections 12 Phase 9 and 13).

A suite is a frozen JSON file of tasks. Every task has an ORACLE: checks the harness runs on the
finished workspace, independent of the `--accept` criteria the agent's own verifier used. That
independence is what makes "the verifier passed but the work is wrong" measurable.
"""
