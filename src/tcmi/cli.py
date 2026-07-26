from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tcmi.config import load_config
from tcmi.data.audit import audit_dataset
from tcmi.data.generator import generate_dataset
from tcmi.evaluation.aggregate import aggregate_results
from tcmi.evaluation.decision import render_decision
from tcmi.evaluation.probe import run_probes
from tcmi.matrix import command_plan, execute_matrix
from tcmi.status import collect_status
from tcmi.training.trainer import train_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcmi",
        description="Task-conditional multimodal information value MVP",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="生成确定性受控数据")
    _add_config_argument(generate)

    audit = subparsers.add_parser("audit", help="审计数据 manifest 与不变量")
    _add_config_argument(audit)

    train = subparsers.add_parser("train", help="执行一个训练 cell")
    _add_config_argument(train)
    train.add_argument("--architecture", required=True)
    train.add_argument("--train-mode", required=True)
    train.add_argument("--condition", required=True)
    train.add_argument("--seed", required=True, type=int)

    probe = subparsers.add_parser("probe", help="执行冻结表示线性 probe")
    _add_config_argument(probe)
    probe.add_argument("--run-dir", required=True, type=Path)
    probe.add_argument("--representation-scope", required=True)

    matrix = subparsers.add_parser("matrix", help="打印或执行预注册矩阵")
    _add_config_argument(matrix)
    mode = matrix.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    matrix.add_argument(
        "--stage",
        choices=("train", "probe", "all"),
        default="all",
        help="--execute 时选择阶段",
    )

    aggregate = subparsers.add_parser("aggregate", help="聚合 seed 与交互效应")
    _add_config_argument(aggregate)

    decide = subparsers.add_parser("decide", help="生成受证据门禁保护的裁决")
    _add_config_argument(decide)

    status = subparsers.add_parser("status", help="查看当前产物状态")
    _add_config_argument(status)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = dispatch(args, config)
    if result is not None:
        if isinstance(result, (dict, list)):
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(result)


def dispatch(args: argparse.Namespace, config: dict[str, Any]) -> Any:
    if args.command == "generate":
        return generate_dataset(config)
    if args.command == "audit":
        return audit_dataset(config)
    if args.command == "train":
        return train_run(
            config,
            architecture=args.architecture,
            train_mode=args.train_mode,
            condition=args.condition,
            seed=args.seed,
        )
    if args.command == "probe":
        return run_probes(
            config,
            run_dir=args.run_dir,
            representation_scope=args.representation_scope,
        )
    if args.command == "matrix":
        if args.print_only:
            return "\n".join(command_plan(args.config, config))
        return execute_matrix(config, stage=args.stage)
    if args.command == "aggregate":
        return aggregate_results(config)
    if args.command == "decide":
        return render_decision(config)
    if args.command == "status":
        return collect_status(config)
    raise RuntimeError(f"未处理的命令: {args.command}")


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)


if __name__ == "__main__":
    main()
