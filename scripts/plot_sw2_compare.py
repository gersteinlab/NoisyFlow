from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple


def _extract_sw2(stats: Dict[str, float]) -> Tuple[float, float, float]:
    return (
        float(stats["sw2_private_ref"]),
        float(stats["sw2_synth_ref"]),
        float(stats["sw2_synth_transported_ref"]),
    )


def _label_for_run(stats: Dict[str, float], cfg, *, fallback: str) -> str:
    stage1_dp = getattr(getattr(cfg, "stage1", None), "dp", None)
    stage2_dp = getattr(getattr(cfg, "stage2", None), "dp", None)
    dp_enabled = bool(getattr(stage1_dp, "enabled", False) or getattr(stage2_dp, "enabled", False))
    if not dp_enabled:
        return fallback

    eps_total = stats.get("epsilon_total_max", None)
    delta = None
    if getattr(stage1_dp, "enabled", False):
        delta = getattr(stage1_dp, "delta", None)
    if delta is None and getattr(stage2_dp, "enabled", False):
        delta = getattr(stage2_dp, "delta", None)

    if eps_total is None:
        return "DP"
    if delta is None:
        return f"DP ($\\varepsilon={float(eps_total):.2f}$)"
    return f"DP ($\\varepsilon={float(eps_total):.2f},\\ \\delta={float(delta):.0e}$)"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(description="Compare SW2 distances for two NoisyFlow configs.")
    parser.add_argument("--config-a", required=True, help="First YAML config path.")
    parser.add_argument("--config-b", required=True, help="Second YAML config path.")
    parser.add_argument("--device", default=None, help="Override device (e.g., cpu, cuda).")
    parser.add_argument("--output", default="sw2_compare.pdf", help="Output path (pdf/png).")
    parser.add_argument("--title", default=None, help="Optional title.")
    parser.add_argument("--label-a", default="Non-private", help="Legend label for config A (if non-DP).")
    parser.add_argument("--label-b", default="DP", help="Legend label for config B (if non-DP).")
    args = parser.parse_args()

    from noisyflow.config import load_config
    from run import run_experiment

    cfg_a = load_config(args.config_a)
    cfg_b = load_config(args.config_b)
    if args.device is not None:
        cfg_a.device = str(args.device)
        cfg_b.device = str(args.device)

    stats_a = run_experiment(cfg_a)
    stats_b = run_experiment(cfg_b)

    a_vals = _extract_sw2(stats_a)
    b_vals = _extract_sw2(stats_b)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting (pip install matplotlib).") from exc

    labels = ["Private vs Ref", "Synth vs Ref", "Transported vs Ref"]
    x = np.arange(len(labels))
    width = 0.36

    label_a = _label_for_run(stats_a, cfg_a, fallback=str(args.label_a))
    label_b = _label_for_run(stats_b, cfg_b, fallback=str(args.label_b))

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(x - width / 2, a_vals, width, label=label_a, color="#4C78A8")
    ax.bar(x + width / 2, b_vals, width, label=label_b, color="#F58518")

    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Sliced W2 distance (lower is better)")
    ax.set_title(args.title or "SW2 to target reference (DP vs non-DP)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    for i, (va, vb) in enumerate(zip(a_vals, b_vals)):
        ax.text(i - width / 2, va, f"{va:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, vb, f"{vb:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Saved plot to {args.output}")
    print("A SW2:", a_vals)
    print("B SW2:", b_vals)


if __name__ == "__main__":
    main()

