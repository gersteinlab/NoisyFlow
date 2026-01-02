from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def _metric_label(metric: str) -> str:
    metric = metric.strip()
    if metric == "acc":
        return "accuracy (synth)"
    if metric == "acc_ref_only":
        return "accuracy (ref only)"
    if metric == "acc_ref_plus_synth":
        return "accuracy (ref+synth)"
    return metric


def _load_points(path: str) -> Tuple[str, List[Tuple[float, float]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metric = str(payload.get("metric", "")).strip()
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Invalid results list in {path}")

    points: List[Tuple[float, float]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        eps = r.get("epsilon", None)
        util = r.get("utility", None)
        if eps is None or util is None:
            continue
        points.append((float(eps), float(util)))

    if not points:
        raise ValueError(f"No valid (epsilon, utility) points in {path}")
    points.sort(key=lambda x: x[0])
    return metric, points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot privacy/utility comparisons for Stage2 schemes from JSON sweep outputs."
    )
    parser.add_argument(
        "--syn",
        default="tex/privacy_curve_stage2_syn_nodp.json",
        help="JSON from synth-only Stage2 (no DP).",
    )
    parser.add_argument(
        "--mixed",
        default="tex/privacy_curve_stage2_mixed_dp.json",
        help="JSON from mixed Stage2 (synth+private with DP).",
    )
    parser.add_argument(
        "--private",
        default="tex/privacy_utility_cellot_ref50_acc_ref_plus_synth.json",
        help="JSON from private-only Stage2 (DP).",
    )
    parser.add_argument(
        "--output",
        default="tex/privacy_curve_stage2_schemes_overlay.pdf",
        help="Output PDF path.",
    )
    parser.add_argument("--png", default=None, help="Optional output PNG path (for quick preview).")
    parser.add_argument(
        "--layout",
        choices=["overlay", "panel"],
        default="overlay",
        help="Plot layout: overlay (single axes) or panel (3 stacked subplots).",
    )
    args = parser.parse_args()

    syn_metric, syn_points = _load_points(args.syn)
    mixed_metric, mixed_points = _load_points(args.mixed)
    priv_metric, priv_points = _load_points(args.private)

    metric = syn_metric or mixed_metric or priv_metric
    if mixed_metric and metric and mixed_metric != metric:
        raise ValueError(f"Metric mismatch: {args.syn} uses {metric}, {args.mixed} uses {mixed_metric}")
    if priv_metric and metric and priv_metric != metric:
        raise ValueError(f"Metric mismatch: {args.syn} uses {metric}, {args.private} uses {priv_metric}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting (pip install matplotlib).") from exc

    all_x = [x for x, _ in (syn_points + mixed_points + priv_points)]
    all_y = [y for _, y in (syn_points + mixed_points + priv_points)]
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)
    pad = 0.02
    y_lo = max(0.0, y_min - pad)
    y_hi = min(1.0, y_max + pad)

    if args.layout == "panel":
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 8), sharex=True, sharey=True)
        panels: List[Tuple[str, List[Tuple[float, float]]]] = [
            ("Stage2: synth only (no DP)", syn_points),
            ("Stage2: synth+private (DP)", mixed_points),
            ("Stage2: private only (DP)", priv_points),
        ]

        for ax, (title, pts) in zip(axes, panels):
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            ax.plot(xs, ys, marker="o")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("epsilon (approx)")
        axes[0].set_ylabel(_metric_label(metric))
        axes[0].set_xlim(0.0, x_max * 1.02)
        axes[0].set_ylim(y_lo, y_hi)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {
            "syn": "#4C78A8",
            "mixed": "#F58518",
            "private": "#54A24B",
        }
        series: List[Tuple[str, List[Tuple[float, float]], str]] = [
            ("Stage2 synth-only (no DP; ε=Stage1)", syn_points, colors["syn"]),
            ("Stage2 mixed (DP; ε=Stage1+Stage2)", mixed_points, colors["mixed"]),
            ("Stage2 private-only (DP; ε=Stage1+Stage2)", priv_points, colors["private"]),
        ]
        for label, pts, color in series:
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            ax.plot(xs, ys, marker="o", color=color, label=label)
        ax.set_xlabel("epsilon (approx)")
        ax.set_ylabel(_metric_label(metric))
        ax.set_xlim(0.0, x_max * 1.02)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")

    if args.png:
        png_path = Path(args.png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=200)
        print(f"Saved PNG preview to {png_path}")


if __name__ == "__main__":
    main()
