#!/usr/bin/env python3
"""Analyze a long-format CSV file with pyGDIS.

Expected CSV columns for the included example::

    parameter,time,x1,x2

The simplest use is to place this script and the CSV in the same directory and
run::

    python csv_data_example.py

Results are written to a new ``gdis_results`` folder beside the input CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gdis import GDIS
from gdis.datasets import trajectories_from_long_dataframe
from gdis.plotting import plot_components, plot_gdis


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "synthetic_parameter_sweep.csv"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Long-format CSV file. Default: synthetic_parameter_sweep.csv beside this script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Default: gdis_results beside the input CSV.",
    )
    parser.add_argument("--parameter-column", default="parameter")
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--state-columns", nargs="+", default=["x1", "x2"])
    parser.add_argument(
        "--critical-value",
        type=float,
        default=None,
        help="Optional known transition parameter. If omitted, pyGDIS estimates it from transition energy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_path.parent / "gdis_results"
    )

    if not input_path.exists():
        raise FileNotFoundError(f"CSV input not found: {input_path}")

    frame = pd.read_csv(input_path)
    trajectories, parameters = trajectories_from_long_dataframe(
        frame,
        parameter_column=args.parameter_column,
        state_columns=args.state_columns,
        time_column=args.time_column,
    )

    result = GDIS().fit_transform(
        trajectories=trajectories,
        parameters=parameters,
        critical_value=args.critical_value,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "gdis_results.csv"
    summary_path = output_dir / "analysis_summary.txt"
    gdis_figure = output_dir / "gdis_profile.png"
    components_figure = output_dir / "gdis_components.png"

    result.to_dataframe().to_csv(results_path, index=False)
    plot_gdis(result, output_path=gdis_figure, dpi=300)
    plot_components(result, output_path=components_figure, dpi=300)
    plt.close("all")

    peak_index = int(result.gdis.argmax())
    transition_index = int(result.components["transition_energy"].argmax())
    summary = [
        "pyGDIS CSV example analysis",
        "=" * 29,
        f"Input file: {input_path.name}",
        f"Rows: {len(frame)}",
        f"Parameter values: {len(parameters)}",
        f"State columns: {', '.join(args.state_columns)}",
        f"Resolved critical value: {result.metadata['critical_value']:.6g}",
        f"Critical-value source: {result.metadata['critical_value_source']}",
        f"Maximum GDIS: {result.gdis[peak_index]:.6f} at parameter {result.parameters[peak_index]:.6g}",
        f"Maximum transition energy at parameter: {result.parameters[transition_index]:.6g}",
        f"Output folder: {output_dir}",
    ]
    summary_text = "\n".join(summary) + "\n"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print("Created:")
    print(f"  {results_path.name}")
    print(f"  {gdis_figure.name}")
    print(f"  {components_figure.name}")
    print(f"  {summary_path.name}")


if __name__ == "__main__":
    main()
