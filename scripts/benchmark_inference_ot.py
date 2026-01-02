from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from noisyflow.data.synthetic import make_federated_mixture_gaussians
from noisyflow.stage1.networks import VelocityField
from noisyflow.stage1.training import sample_flow_euler
from noisyflow.stage2.networks import ICNN
from noisyflow.utils import set_seed


@dataclass(frozen=True)
class BenchResult:
    d: int
    sample_ms: float
    transport_ms_framework: float
    transport_ms_sinkhorn: float
    total_ms_framework: float
    total_ms_sinkhorn: float

    @property
    def speedup_transport(self) -> float:
        return self.transport_ms_sinkhorn / max(1e-9, self.transport_ms_framework)

    @property
    def speedup_total(self) -> float:
        return self.total_ms_sinkhorn / max(1e-9, self.total_ms_framework)


def _percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0,1]")
    xs_sorted = sorted(xs)
    idx = int(round(q * (len(xs_sorted) - 1)))
    return float(xs_sorted[idx])


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _time_op_ms(
    device: torch.device,
    op: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> Tuple[float, torch.Tensor]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be >= 0 and repeats must be > 0")

    if device.type == "cuda":
        _sync_if_cuda(device)
        last_out: Optional[torch.Tensor] = None
        for _ in range(warmup):
            last_out = op()
        _sync_if_cuda(device)

        times_ms: List[float] = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            last_out = op()
            end.record()
            end.synchronize()
            times_ms.append(float(start.elapsed_time(end)))
        assert last_out is not None
        return float(statistics.median(times_ms)), last_out

    last_out = None
    for _ in range(warmup):
        last_out = op()
    times_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        last_out = op()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    assert last_out is not None
    return float(statistics.median(times_ms)), last_out


@torch.no_grad()
def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).transpose(0, 1)
    cost = x2 + y2 - 2.0 * (x @ y.transpose(0, 1))
    return torch.clamp(cost, min=0.0)


@torch.no_grad()
def sinkhorn_barycentric_projection(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    n_iters: int,
    cost_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Balanced entropic OT (Sinkhorn) + barycentric projection.

    source: (n, d)
    target: (m, d)
    returns: (n, d) transported points
    """
    if source.dim() != 2 or target.dim() != 2:
        raise ValueError("source and target must be 2D tensors")
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"dimension mismatch: {source.shape} vs {target.shape}")
    if n_iters <= 0:
        raise ValueError("n_iters must be > 0")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    n = int(source.shape[0])
    m = int(target.shape[0])
    if n == 0 or m == 0:
        raise ValueError("source and target must be non-empty")

    cost = _pairwise_sq_dist(source, target)
    if cost_scale is not None:
        cost = cost / float(cost_scale)

    # Uniform weights.
    a = torch.full((n,), 1.0 / float(n), device=source.device, dtype=source.dtype)
    b = torch.full((m,), 1.0 / float(m), device=source.device, dtype=source.dtype)

    # Standard Sinkhorn scaling on K = exp(-C/eps).
    K = torch.exp(-cost / float(epsilon))
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    tiny = torch.finfo(source.dtype).tiny

    for _ in range(n_iters):
        Kv = K @ v
        u = a / torch.clamp(Kv, min=tiny)
        KTu = K.transpose(0, 1) @ u
        v = b / torch.clamp(KTu, min=tiny)

    P = (u[:, None] * K) * v[None, :]
    row_sums = P.sum(dim=1, keepdim=True)
    transported = (P @ target) / torch.clamp(row_sums, min=tiny)
    return transported


def _parse_int_list(values: Iterable[str]) -> List[int]:
    out: List[int] = []
    for v in values:
        if not v:
            continue
        out.append(int(v))
    return out


def run_benchmark(
    *,
    dims: Sequence[int],
    n: int,
    flow_steps: int,
    sinkhorn_epsilon: float,
    sinkhorn_iters: int,
    repeats: int,
    warmup: int,
    seed: int,
    device: str,
    flow_hidden: Sequence[int],
    ot_hidden: Sequence[int],
    dtype: torch.dtype,
) -> List[BenchResult]:
    set_seed(seed)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu or fix CUDA setup.")

    dev = torch.device(device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    results: List[BenchResult] = []
    for d in dims:
        # Gaussian mixture w/ domain shift for target samples.
        client_datasets, target_ref, _target_test = make_federated_mixture_gaussians(
            K=1,
            n_per_client=max(1, int(n)),
            n_target_ref=max(1, int(n)),
            n_target_test=1,
            d=int(d),
            num_classes=1,
            seed=seed,
        )
        target = target_ref.tensors[0][:n].to(device=dev, dtype=dtype)

        flow = VelocityField(
            d=int(d),
            num_classes=1,
            hidden=list(flow_hidden),
            time_emb_dim=64,
            label_emb_dim=16,
            act="silu",
        ).to(device=dev, dtype=dtype)
        flow.eval()

        ot = ICNN(
            d=int(d),
            hidden=list(ot_hidden),
            act="relu",
            add_strong_convexity=0.1,
        ).to(device=dev, dtype=dtype)
        ot.eval()

        labels = torch.zeros((n,), device=dev, dtype=torch.long)

        def op_sample() -> torch.Tensor:
            return sample_flow_euler(flow, labels, n_steps=int(flow_steps))

        sample_ms, x_syn = _time_op_ms(dev, op_sample, warmup=warmup, repeats=repeats)

        def op_transport_framework() -> torch.Tensor:
            # ICNN.transport uses autograd internally (Brenier map).
            return ot.transport(x_syn)

        transport_fw_ms, y_fw = _time_op_ms(dev, op_transport_framework, warmup=warmup, repeats=repeats)

        def op_transport_sinkhorn() -> torch.Tensor:
            return sinkhorn_barycentric_projection(
                x_syn,
                target,
                epsilon=float(sinkhorn_epsilon),
                n_iters=int(sinkhorn_iters),
                cost_scale=float(d),
            )

        transport_sh_ms, y_sh = _time_op_ms(dev, op_transport_sinkhorn, warmup=warmup, repeats=repeats)

        # Ensure ops were executed (avoid accidental dead-code elimination).
        _ = float(y_fw.mean().detach().float().cpu().item()) + float(y_sh.mean().detach().float().cpu().item())

        results.append(
            BenchResult(
                d=int(d),
                sample_ms=float(sample_ms),
                transport_ms_framework=float(transport_fw_ms),
                transport_ms_sinkhorn=float(transport_sh_ms),
                total_ms_framework=float(sample_ms + transport_fw_ms),
                total_ms_sinkhorn=float(sample_ms + transport_sh_ms),
            )
        )
    return results


def _format_table(rows: Sequence[BenchResult]) -> str:
    headers = [
        "d",
        "sample_ms",
        "transport_fw_ms",
        "transport_sinkhorn_ms",
        "total_fw_ms",
        "total_sinkhorn_ms",
        "speedup_transport",
        "speedup_total",
    ]
    data = [
        [
            str(r.d),
            f"{r.sample_ms:.2f}",
            f"{r.transport_ms_framework:.2f}",
            f"{r.transport_ms_sinkhorn:.2f}",
            f"{r.total_ms_framework:.2f}",
            f"{r.total_ms_sinkhorn:.2f}",
            f"{r.speedup_transport:.2f}x",
            f"{r.speedup_total:.2f}x",
        ]
        for r in rows
    ]
    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = []
    lines.append("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in data:
        lines.append("  ".join(row[i].rjust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NoisyFlow OT inference vs Sinkhorn (GPU).")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dims", type=str, nargs="*", default=["512", "1024", "2048", "4096"])
    parser.add_argument("--n", type=int, default=500, help="Number of source and target points (default: 500).")
    parser.add_argument("--flow-steps", type=int, default=50)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1.0)
    parser.add_argument("--sinkhorn-iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flow-hidden", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--ot-hidden", type=int, nargs="*", default=[128, 128])
    args = parser.parse_args()

    dims = _parse_int_list(args.dims)
    if not dims:
        raise SystemExit("No dims provided")

    dtype = torch.float32
    results = run_benchmark(
        dims=dims,
        n=int(args.n),
        flow_steps=int(args.flow_steps),
        sinkhorn_epsilon=float(args.sinkhorn_epsilon),
        sinkhorn_iters=int(args.sinkhorn_iters),
        repeats=int(args.repeats),
        warmup=int(args.warmup),
        seed=int(args.seed),
        device=str(args.device),
        flow_hidden=list(args.flow_hidden),
        ot_hidden=list(args.ot_hidden),
        dtype=dtype,
    )

    print(
        f"device={args.device}  n={args.n}  flow_steps={args.flow_steps}  sinkhorn_eps={args.sinkhorn_epsilon}  sinkhorn_iters={args.sinkhorn_iters}  repeats={args.repeats}"
    )
    if args.device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(_format_table(results))

    best = max(results, key=lambda r: r.speedup_total)
    worst = min(results, key=lambda r: r.speedup_total)
    print(
        f"\nSpeedup (total) range across dims: {worst.speedup_total:.2f}x .. {best.speedup_total:.2f}x"
    )


if __name__ == "__main__":
    main()
