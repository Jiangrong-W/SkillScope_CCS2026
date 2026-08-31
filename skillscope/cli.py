from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .common.config import AppConfig
from .pipeline import AnalyzePipeline, RepairPipeline, ValidatePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SkillScope command line interface")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in ("analyze", "validate", "repair"):
        command = subparsers.add_parser(command_name)
        command.add_argument("skill_path", type=Path, help="Path to the target skill bundle directory")
        command.add_argument(
            "--artifacts-root",
            type=Path,
            default=None,
            help="Optional directory for pipeline artifacts. Defaults to ./artifacts.",
        )
        if command_name in {"validate", "repair"}:
            command.add_argument(
                "--validation-mode",
                choices=("dynamic", "static"),
                default=None,
                help=(
                    "Choose dynamic replay-based over-privilege validation or "
                    "static reasoning. Repair requires dynamic validation; "
                    "static mode is available only for validate. Defaults to "
                    "dynamic."
                ),
            )
            command.add_argument(
                "--user-prompt",
                action="append",
                default=[],
                help=(
                    "Concrete user task to validate. Repeat this option to provide multiple task contexts; "
                    "graph-generated representative tasks are still included."
                ),
            )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    config = AppConfig.from_project_root(
        project_root=project_root,
        artifact_root=args.artifacts_root or (Path.cwd() / "artifacts"),
        validation_mode=getattr(args, "validation_mode", None),
    )
    if args.command == "repair" and config.validation_mode != "dynamic":
        parser.error(
            "repair requires dynamic replay validation; static validation is "
            "available only as a predictive fallback for `skillscope validate`"
        )
    config.artifact_root.mkdir(parents=True, exist_ok=True)

    if args.command == "analyze":
        summary = AnalyzePipeline(config).run(args.skill_path)
    elif args.command == "validate":
        summary = ValidatePipeline(config).run(args.skill_path, user_prompts=args.user_prompt)
    else:
        summary = RepairPipeline(config).run(args.skill_path, user_prompts=args.user_prompt)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
