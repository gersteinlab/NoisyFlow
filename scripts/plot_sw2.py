from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional


def _extract_sw2(stats: Dict[str, float]) -> Dict[str, float]:
    keys = [
        "sw2_private_ref",
        "sw2_synth_ref",
        "sw2_synth_transported_ref",
    ]
    missing = [k for k in keys if k not in stats]
    if missing:
        raise KeyError(f"Missing SW2 keys in stats: {missing}")
    return {k: float(stats[k]) for k in keys}


def _format_privacy_text(stats: Dict[str, float], cfg) -> str:
    eps_total = stats.get("epsilon_total_max", None)
    eps_flow = stats.get("epsilon_flow_max", None)
    eps_ot = stats.get("epsilon_ot_max", None)

    stage1_dp = getattr(getattr(cfg, "stage1", None), "dp", None)
    stage2_dp = getattr(getattr(cfg, "stage2", None), "dp", None)
    dp_enabled = bool(getattr(stage1_dp, "enabled", False) or getattr(stage2_dp, "enabled", False))

    deltas = []
    if getattr(stage1_dp, "enabled", False):
        deltas.append(getattr(stage1_dp, "delta", None))
    if getattr(stage2_dp, "enabled", False):
        deltas.append(getattr(stage2_dp, "delta", None))
    delta: Optional[float] = None
    for d in deltas:
        if d is None:
            continue
        delta = float(d)
        break

    if not dp_enabled:
        return "Non-private"

    parts = []
    if eps_total is not None:
        parts.append("eps_total=" + f"{float(eps_total):.3f}")
    else:
        if eps_flow is not None:
            parts.append("eps_flow=" + f"{float(eps_flow):.3f}")
        if eps_ot is not None:
            parts.append("eps_ot=" + f"{float(eps_ot):.3f}")
    if delta is not None:
        parts.append("delta=" + f"{delta:.0e}")
    return "DP: " + ", ".join(parts) if parts else "DP"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(description="Plot SW2 distances from a NoisyFlow run.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output", default="sw2_bar.png", help="Output PNG path.")
    parser.add_argument("--device", default=None, help="Override device (e.g., cpu, cuda).")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    args = parser.parse_args()

    from noisyflow.config import load_config
    from run import run_experiment

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.device = str(args.device)

    stats = run_experiment(cfg)
    sw2 = _extract_sw2(stats)
    privacy_text = _format_privacy_text(stats, cfg)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting (pip install matplotlib).") from exc

    labels = ["Private vs Ref", "Synth vs Ref", "Transported vs Ref"]
    values = [
        sw2["sw2_private_ref"],
        sw2["sw2_synth_ref"],
        sw2["sw2_synth_transported_ref"],
    ]

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.bar(range(len(values)), values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_xticks(range(len(values)), labels, rotation=15, ha="right")
    ax.set_ylabel("Sliced W2 distance (lower is better)")
    ax.set_title(args.title or "Sliced W2 to target reference")
    ax.text(
        0.99,
        0.98,
        privacy_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "boxstyle": "round,pad=0.2"},
    )
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)

    print(f"Saved plot to {args.output}")
    print("SW2:", sw2)


if __name__ == "__main__":
    main()
