import torch
from typing import Optional

def _pairwise_sq_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute ‖a_i‑b_j‖² without forming (m,m,d) tensors (GPU‑friendly)."""
    # a: (m,d), b: (n,d) (contiguous not required)
    a_norm = (a ** 2).sum(-1, keepdim=True)          # (m,1)
    b_norm = (b ** 2).sum(-1, keepdim=True).T         # (1,n)
    # dist² = |a|² + |b|² − 2a·b
    return a_norm + b_norm - 2 * (a @ b.T)            # (m,n)

def _rbf_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    sigma: float,
    *,
    chunk: Optional[int] = None,
) -> torch.Tensor:
    """RBF kernel K exp(-‖x-y‖²/2σ²) with optional chunked computation."""
    if chunk is None or a.size(0) <= chunk:
        d2 = _pairwise_sq_dists(a, b)
        return torch.exp(-d2 / (2 * sigma ** 2))
    # Block‑wise over rows of *a* to keep peak memory small
    res_rows = []
    for start in range(0, a.size(0), chunk):
        end = min(start + chunk, a.size(0))
        d2_block = _pairwise_sq_dists(a[start:end], b)
        res_rows.append(torch.exp(-d2_block / (2 * sigma ** 2)))
    return torch.cat(res_rows, dim=0)

def _median_heuristic(x: torch.Tensor, max_samples: int = 1000) -> float:
    """Return median pairwise distance (√) for RBF bandwidth heuristic."""
    m = x.size(0)
    if m > max_samples:
        idx = torch.randperm(m, device=x.device)[:max_samples]
        x = x[idx]
    d2 = _pairwise_sq_dists(x, x)
    i, j = torch.triu_indices(d2.size(0), d2.size(1), offset=1, device=x.device)
    median = torch.median(d2[i, j]).sqrt().item()
    return max(median, 1e-6)  # avoid zero

