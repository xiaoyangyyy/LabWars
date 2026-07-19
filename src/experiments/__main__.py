"""CLI entry points for Part 4 experiments."""

from __future__ import annotations

import argparse
import sys

from src.experiments.aggregate import aggregate_validity_gate, write_aggregate_report
from src.experiments.batch import run_batch, run_full_matrix
from src.experiments.conditions import list_conditions
from src.experiments.report import generate_report
from src.experiments.runner import run_single
from src.experiments.scale import run_scale_experiment
from src.experiments.scientific_protocol import run_scientific_protocol


def cmd_run(args: argparse.Namespace) -> None:
    result = run_single(args.experiment, args.seed, args.condition, max_rounds=args.rounds)
    log = result["log"]
    print(f"run_id={log.run_id} rounds={len(log.round_records)}")
    for key in result["condition"].primary_outcomes:
        print(f"  {key}={log.outcomes.get(key)}")


def cmd_batch(args: argparse.Namespace) -> None:
    if args.experiment.upper() == "ALL":
        run_full_matrix(seeds=args.seeds, parallel=args.parallel, output_dir=args.output)
        print("Full matrix complete.")
        return
    rows = run_batch(
        args.experiment,
        seeds=args.seeds,
        parallel=args.parallel,
        output_dir=args.output,
        max_rounds=args.rounds,
    )
    print(f"Batch {args.experiment}: {len(rows)} runs completed.")


def cmd_report(args: argparse.Namespace) -> None:
    path = generate_report(
        run_id=args.run_id,
        experiment_id=args.experiment,
        condition_id=args.condition,
        seed=args.seed,
        output_dir=args.output,
    )
    print(path)


def cmd_aggregate(args: argparse.Namespace) -> None:
    if args.validity:
        gate = aggregate_validity_gate(list(range(args.seeds)))
        print(gate)
        return
    path = write_aggregate_report(args.experiment, output_dir=args.output)
    print(path)




def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def cmd_scale(args: argparse.Namespace) -> None:
    result = run_scale_experiment(
        population_sizes=_parse_int_list(args.population_sizes),
        rounds=args.rounds,
        seeds=list(range(args.seeds)),
        policy_mode=args.policy_mode,
        llm_provider=args.llm_provider,
        population_labs=args.population_labs,
        output_dir=args.output,
        write_output=True,
    )
    print(result.summary)



def cmd_protocol(args: argparse.Namespace) -> None:
    result = run_scientific_protocol(
        protocol_id=args.protocol_id,
        population_sizes=_parse_int_list(args.population_sizes),
        rounds=args.rounds,
        seeds=list(range(args.seeds)),
        conditions=[part.strip() for part in args.conditions.split(",") if part.strip()],
        policy_mode=args.policy_mode,
        llm_provider=args.llm_provider,
        output_dir=args.output,
        write_output=True,
    )
    print(result.summary)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="labwars-experiments", description="LabWars Part 4 experiment CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a single experiment condition")
    p_run.add_argument("--experiment", "-e", required=True, help="A|B|C|D|V")
    p_run.add_argument("--condition", "-c", default=None, help="Condition id e.g. A1")
    p_run.add_argument("--seed", "-s", type=int, default=42)
    p_run.add_argument("--rounds", type=int, default=60)
    p_run.set_defaults(func=cmd_run)

    p_batch = sub.add_parser("batch", help="Batch run all conditions for an experiment")
    p_batch.add_argument("--experiment", "-e", required=True, help="A|B|C|D|V|ALL")
    p_batch.add_argument("--seeds", type=int, default=30)
    p_batch.add_argument("--parallel", type=int, default=1)
    p_batch.add_argument("--rounds", type=int, default=60)
    p_batch.add_argument("--output", "-o", default=None)
    p_batch.set_defaults(func=cmd_batch)

    p_report = sub.add_parser("report", help="Generate decompilation report")
    p_report.add_argument("--run-id", default=None)
    p_report.add_argument("--experiment", "-e", default="A")
    p_report.add_argument("--condition", "-c", default="A1")
    p_report.add_argument("--seed", "-s", type=int, default=42)
    p_report.add_argument("--output", "-o", default=None)
    p_report.set_defaults(func=cmd_report)

    p_agg = sub.add_parser("aggregate", help="Aggregate batch results")
    p_agg.add_argument("--experiment", "-e", default="A")
    p_agg.add_argument("--output", "-o", default=None)
    p_agg.add_argument("--validity", action="store_true")
    p_agg.add_argument("--seeds", type=int, default=10)
    p_agg.set_defaults(func=cmd_aggregate)
    p_scale = sub.add_parser("scale", help="Run population-size scale benchmark")
    p_scale.add_argument("--population-sizes", default="14,50,100,200", help="Comma-separated sizes, e.g. 14,50,100,200")
    p_scale.add_argument("--rounds", type=int, default=100)
    p_scale.add_argument("--seeds", type=int, default=3)
    p_scale.add_argument("--policy-mode", default="social_physics", choices=["social_physics", "dual_engine", "llm_native"])
    p_scale.add_argument("--llm-provider", default="scripted")
    p_scale.add_argument("--population-labs", type=int, default=None)
    p_scale.add_argument("--output", "-o", default=None)
    p_scale.set_defaults(func=cmd_scale)
    p_protocol = sub.add_parser("protocol", help="Run repeatable experimental-science protocol")
    p_protocol.add_argument("--protocol-id", default="agent_social_dynamics_protocol_v1")
    p_protocol.add_argument("--population-sizes", default="10,50,100,500", help="Comma-separated sizes")
    p_protocol.add_argument("--rounds", type=int, default=500)
    p_protocol.add_argument("--seeds", type=int, default=100)
    p_protocol.add_argument("--conditions", default="baseline,no_memory,no_status,no_trust,no_hierarchy")
    p_protocol.add_argument("--policy-mode", default="social_physics", choices=["social_physics", "dual_engine", "llm_native"])
    p_protocol.add_argument("--llm-provider", default="scripted")
    p_protocol.add_argument("--output", "-o", default=None)
    p_protocol.set_defaults(func=cmd_protocol)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "condition", None) is None and hasattr(args, "experiment"):
        try:
            args.condition = list_conditions(args.experiment)[0]
        except (ValueError, KeyError):
            pass
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
