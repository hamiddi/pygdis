from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .datasets import trajectories_from_long_dataframe
from .plotting import plot_components, plot_gdis
from .score import GDIS
from .systems import ChenSystem, LogisticMap, LorenzSystem, RosslerSystem


def analyze_command(args):
    dataframe = pd.read_csv(args.csv)
    trajectories, parameters = trajectories_from_long_dataframe(
        dataframe, args.parameter, args.state_columns, args.time
    )
    result = GDIS().fit_transform(trajectories, parameters)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result.to_dataframe().to_csv(output / "gdis_results.csv", index=False)
    plot_gdis(result, output_path=str(output / "gdis.png"))
    plot_components(result, output_path=str(output / "components.png"))


def benchmark_command(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for system in [LorenzSystem(), RosslerSystem(), ChenSystem(), LogisticMap()]:
        trajectories, parameters = system.generate_sweep()
        result = GDIS().fit_transform(
            trajectories,
            parameters,
            jacobian_function=getattr(system, "jacobian", None),
            critical_value=system.critical_value,
        )
        system_output = output / system.name.lower()
        system_output.mkdir(parents=True, exist_ok=True)
        result.to_dataframe().to_csv(system_output / "gdis_results.csv", index=False)
        plot_gdis(result, output_path=str(system_output / "gdis.png"))


def build_parser():
    parser = argparse.ArgumentParser(prog="gdis", description="Generalized Dynamical Instability Score")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="Analyze a long-form CSV file.")
    analyze.add_argument("csv")
    analyze.add_argument("--parameter", required=True)
    analyze.add_argument("--state-columns", nargs="+", required=True)
    analyze.add_argument("--time")
    analyze.add_argument("--output", default="gdis_results")
    analyze.set_defaults(func=analyze_command)
    benchmark = subparsers.add_parser("benchmark", help="Run built-in benchmark systems.")
    benchmark.add_argument("--output", default="gdis_benchmarks")
    benchmark.set_defaults(func=benchmark_command)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
