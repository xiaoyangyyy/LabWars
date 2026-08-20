"""CLI entry points for Part 4 experiments."""

from __future__ import annotations

import argparse
import sys

from src.experiments.aggregate import aggregate_validity_gate, write_aggregate_report
from src.experiments.batch import run_batch, run_full_matrix
from src.experiments.conditions import list_conditions
from src.engine.simulation import SimConfig
from src.experiments.causal_mri import run_causal_mri
from src.experiments.paper_protocol import run_paper_protocol
from src.engine.causal.twin import load_factual, sim_config_from_log
from src.experiments.report import generate_report
from src.experiments.runner import run_single
from src.experiments.scale import run_scale_experiment
from src.experiments.scientific_protocol import run_scientific_protocol
from src.experiments.policy_protocol import run_policy_comparison_protocol
from src.experiments.sampling_frontier import run_sampling_frontier
from src.experiments.emergence_challenge import run_egalitarian_emergence_challenge


def cmd_run(args: argparse.Namespace) -> None:
    result = run_single(args.experiment, args.seed, args.condition, max_rounds=args.rounds)
    log = result["log"]
    print(f"run_id={log.run_id} rounds={len(log.round_records)}")
    for key in result["condition"].primary_outcomes:
        print(f"  {key}={log.outcomes.get(key)}")


def cmd_batch(args: argparse.Namespace) -> None:
    if args.experiment.upper() == "ALL":
        run_full_matrix(
            seeds=args.seeds,
            parallel=args.parallel,
            output_dir=args.output,
            skip_existing=args.skip_existing,
        )
        print("Full matrix complete.")
        return
    rows = run_batch(
        args.experiment,
        seeds=args.seeds,
        parallel=args.parallel,
        output_dir=args.output,
        max_rounds=args.rounds,
        skip_existing=args.skip_existing,
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



def cmd_policy_compare(args: argparse.Namespace) -> None:
    result = run_policy_comparison_protocol(
        population_size=args.population_size,
        rounds=args.rounds,
        seeds=list(range(args.seeds)),
        regimes=[part.strip() for part in args.regimes.split(",") if part.strip()],
        llm_provider=args.llm_provider,
        sampled_top_k=args.sampled_top_k,
        output_dir=args.output,
        write_output=True,
    )
    print(result.summary)



def _parse_k_list(raw: str) -> list[int | str]:
    values: list[int | str] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if item.lower() in {"full", "all"}:
            values.append("full")
        else:
            values.append(int(item))
    return values


def cmd_sampling_frontier(args: argparse.Namespace) -> None:
    result = run_sampling_frontier(
        population_size=args.population_size,
        rounds=args.rounds,
        seeds=list(range(args.seeds)),
        k_values=_parse_k_list(args.k_values),
        llm_provider=args.llm_provider,
        output_dir=args.output,
        write_output=True,
    )
    print(result.summary)


def cmd_decompile(args: argparse.Namespace) -> None:
    memory_rounds = _parse_int_list(args.memory_rounds) if args.memory_rounds else []
    factual = load_factual(args.from_jsonl) if getattr(args, "from_jsonl", None) else None
    if factual is not None:
        cfg = sim_config_from_log(factual)
    else:
        cfg = SimConfig(
            max_rounds=args.rounds,
            seed=args.seed,
            mvp=not args.full_cast,
            interventions=[],
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            cognitive_sampling_top_k=args.sampled_top_k,
            policy_mode=args.policy_mode,
        )
    result = run_causal_mri(
        cfg,
        outcome=args.outcome,
        extra_ops=None,
        memory_rounds=memory_rounds or None,
        blame_limit=args.blame_first,
        include_toy_shapley=True,
        auto_battery=bool(getattr(args, "auto_battery", False)),
        factual=factual,
        write_output=True,
        output_dir=args.output,
    )
    print(result["summary"])
    if result.get("json_path"):
        print(result["json_path"])


def cmd_paper(args: argparse.Namespace) -> None:
    factual_path = args.from_jsonl or None
    cfg = None
    if not factual_path:
        cfg = SimConfig(
            max_rounds=args.rounds,
            seed=args.seed,
            mvp=not args.full_cast,
            interventions=[],
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            policy_mode=args.policy_mode,
            cognitive_sampling_top_k=args.sampled_top_k,
        )
    contrasts = [part.strip() for part in (args.contrasts or "").split(",") if part.strip()]
    result = run_paper_protocol(
        cfg,
        from_jsonl=factual_path,
        outcome=args.outcome,
        auto_battery=not args.lite,
        include_lambda=args.include_lambda,
        contrasts=contrasts or None,
        contrast_seeds=[args.seed],
        write_output=True,
        output_dir=args.output,
    )
    print(result.summary)
    if result.markdown_path:
        print(result.markdown_path)



def cmd_egalitarian_challenge(args: argparse.Namespace) -> None:
    result = run_egalitarian_emergence_challenge(
        population_size=args.population_size,
        rounds=args.rounds,
        seeds=list(range(args.seeds)),
        llm_provider=args.llm_provider,
        policy_mode=args.policy_mode,
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
    p_batch.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing run_*.jsonl files and rebuild summary rows instead of rerunning",
    )
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
    p_protocol.add_argument("--conditions", default="baseline,no_memory,no_status,no_trust,no_hierarchy,no_observation")
    p_protocol.add_argument("--policy-mode", default="social_physics", choices=["social_physics", "dual_engine", "llm_native"])
    p_protocol.add_argument("--llm-provider", default="scripted")
    p_protocol.add_argument("--output", "-o", default=None)
    p_protocol.set_defaults(func=cmd_protocol)
    p_policy = sub.add_parser("policy-compare", help="Compare rule, LLM-native, hybrid, and sampled-hybrid policies")
    p_policy.add_argument("--population-size", type=int, default=50)
    p_policy.add_argument("--rounds", type=int, default=60)
    p_policy.add_argument("--seeds", type=int, default=3)
    p_policy.add_argument("--regimes", default="rule_baseline,llm_native,hybrid,hybrid_sampled")
    p_policy.add_argument("--llm-provider", default="scripted")
    p_policy.add_argument("--sampled-top-k", type=int, default=20)
    p_policy.add_argument("--output", "-o", default=None)
    p_policy.set_defaults(func=cmd_policy_compare)
    p_frontier = sub.add_parser("sampling-frontier", help="Sweep cognitive sampling k and report compute/emergence frontier")
    p_frontier.add_argument("--population-size", type=int, default=100)
    p_frontier.add_argument("--rounds", type=int, default=100)
    p_frontier.add_argument("--seeds", type=int, default=3)
    p_frontier.add_argument("--k-values", default="0,5,10,20,50,100,full")
    p_frontier.add_argument("--llm-provider", default="scripted")
    p_frontier.add_argument("--output", "-o", default=None)
    p_frontier.set_defaults(func=cmd_sampling_frontier)

    p_equal = sub.add_parser("egalitarian-challenge", help="Test emergence from equal initial capability/resource/status/network")
    p_equal.add_argument("--population-size", type=int, default=500)
    p_equal.add_argument("--rounds", type=int, default=500)
    p_equal.add_argument("--seeds", type=int, default=10)
    p_equal.add_argument("--llm-provider", default="scripted")
    p_equal.add_argument("--policy-mode", default="social_physics", choices=["social_physics", "dual_engine", "llm_native"])
    p_equal.add_argument("--output", "-o", default=None)
    p_equal.set_defaults(func=cmd_egalitarian_challenge)

    p_mri = sub.add_parser("decompile", help="Paper default: Causal Decompiler MRI on a factual trajectory")
    p_mri.add_argument("--rounds", type=int, default=8)
    p_mri.add_argument("--seed", "-s", type=int, default=11)
    p_mri.add_argument("--mvp", action="store_true", default=True)
    p_mri.add_argument("--full-cast", action="store_true", help="Use the 14-agent story instead of MVP")
    p_mri.add_argument("--llm-provider", default="scripted")
    p_mri.add_argument("--llm-model", default=None)
    p_mri.add_argument("--policy-mode", default="dual_engine", choices=["social_physics", "dual_engine", "llm_native"])
    p_mri.add_argument("--sampled-top-k", type=int, default=None)
    p_mri.add_argument("--memory-rounds", default="", help="Comma-separated delete times for the memory IRF")
    p_mri.add_argument("--blame-first", type=int, default=1, help="How many early events to skip-contrast; 0 disables")
    p_mri.add_argument("--outcome", default="protest_authorship")
    p_mri.add_argument("--output", "-o", default=None)
    p_mri.add_argument("--from-jsonl", default=None, help="Replay MRI from a persisted factual jsonl")
    p_mri.add_argument("--auto-battery", action="store_true", help="IRF + story Shapley + three-worlds")
    p_mri.set_defaults(func=cmd_decompile)

    p_paper = sub.add_parser("paper", help="Top-venue Causal Decompiler protocol (tables + optional CRN contrasts)")
    p_paper.add_argument("--rounds", type=int, default=8)
    p_paper.add_argument("--seed", "-s", type=int, default=11)
    p_paper.add_argument("--mvp", action="store_true", default=True)
    p_paper.add_argument("--full-cast", action="store_true")
    p_paper.add_argument("--llm-provider", default="scripted")
    p_paper.add_argument("--llm-model", default=None)
    p_paper.add_argument("--policy-mode", default="dual_engine", choices=["social_physics", "dual_engine", "llm_native"])
    p_paper.add_argument("--sampled-top-k", type=int, default=None)
    p_paper.add_argument("--outcome", default="protest_authorship")
    p_paper.add_argument("--from-jsonl", default=None)
    p_paper.add_argument("--contrasts", default="", help="Comma-separated experiments, e.g. A or A,C")
    p_paper.add_argument("--include-lambda", action="store_true")
    p_paper.add_argument("--lite", action="store_true", help="Identity + split-Y + toy Shapley only")
    p_paper.add_argument("--output", "-o", default=None)
    p_paper.set_defaults(func=cmd_paper)

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
