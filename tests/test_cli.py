from __future__ import annotations

from tcmi.cli import build_parser


def test_cli_exposes_all_mvp_commands() -> None:
    parser = build_parser()
    subparsers_action = next(
        action for action in parser._actions if action.dest == "command"
    )
    assert set(subparsers_action.choices) == {
        "generate",
        "audit",
        "train",
        "probe",
        "matrix",
        "aggregate",
        "decide",
        "status",
    }
