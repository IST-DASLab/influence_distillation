import torch
from infdist.utils import tuple_utils
from trak.projectors import BasicProjector, CudaProjector, ProjectionType

def get_trak_projector_cls(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        num_sms = torch.cuda.get_device_properties(
            device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(
            8, 1_000, device=device), 512, 0, num_sms)
        projector_cls = CudaProjector
        print("Using CudaProjector")
    except:
        projector_cls = BasicProjector
        print("Using BasicProjector")
    return projector_cls

def _project_grad(projector, grads_tuple, num_parts):
    if projector is None:
        # just flatten the tuple
        return tuple_utils.flatten(grads_tuple)

    if isinstance(projector, (BasicProjector, CudaProjector)):
        # we break it in num_parts parts to avoid memory issues, as we have to do it in fp32
        dtype = grads_tuple[0].dtype
        proj_dtype = getattr(projector, 'dtype', torch.float32)
        
        part_len_approx = len(grads_tuple) // num_parts
        remainder = len(grads_tuple) % num_parts
        part_lens = [part_len_approx] * num_parts
        for i in range(remainder):
            part_lens[i] += 1
        assert sum(part_lens) == len(grads_tuple)
        
        out = []
        for i, part_len in enumerate(part_lens):
            if part_len == 0:
                continue
            start = sum(part_lens[:i])
            end = start + part_len
            part_grads = grads_tuple[start:end]
            part_grads_flat = tuple_utils.flatten(part_grads)
            part_proj = (projector.project(part_grads_flat.unsqueeze(0).to(proj_dtype), model_id=i).squeeze() / projector.proj_dim ** 0.5).to(dtype)
            out.append(part_proj)
            del part_proj

        return torch.cat(out)
    
    elif isinstance(projector, Had1GProjector):
        assert num_parts == 1
        if isinstance(grads_tuple, tuple):
            grads_tuple = tuple_utils.flatten(grads_tuple)
        return projector.project(grads_tuple)

    else:
        raise ValueError(f"Unknown projector type: {type(projector)}")

def _cosntruct_projector(full_dim, proj_dim, seed, device, dtype, block_size=128, proj_type='rad'):
    if proj_type == 'rad':
        projector_cls = get_trak_projector_cls(device)
        return projector_cls(
            grad_dim=full_dim,
            proj_dim=proj_dim,
            seed=seed,
            proj_type=ProjectionType.rademacher,
            device=device,
            dtype=dtype,
            block_size=block_size,
            max_batch_size=16
        )
    elif proj_type == 'had':
        return Had1GProjector(
            full_dim=full_dim,
            proj_dim=proj_dim,
            seed=seed,
            device=device,
            dtype=dtype
        )
    else:
        raise ValueError(f"Unknown projection type: {proj_type}")




import math
from fast_hadamard_transform import hadamard_transform   # pip install fast-hadamard-transform

BLOCK  = 32_768                         # 2^15
PAD_TO = BLOCK * BLOCK                  # 2^30  (≈1.07 G elements)

class Had1GProjector:
    def __init__(self, full_dim, proj_dim, seed, device, dtype):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        self.n = full_dim
        assert self.n <= PAD_TO
        self.k = proj_dim
        self.sign = torch.randint(0, 2, (PAD_TO,), device=device, dtype=dtype, generator=g).mul_(2).sub_(1)
        self.idxs  = torch.randperm(full_dim, device=device)[:self.k]
    
    @torch.no_grad()
    def project(self, x):
        assert x.is_cuda and x.ndim == 1,  "pass a 1-D CUDA tensor"
        n_orig = x.numel()
        assert n_orig == self.n

        # 1. pad to exactly 2^30
        if n_orig < PAD_TO:
            x = torch.cat([x, x.new_zeros(PAD_TO - n_orig)], 0)

        # 2. random ±1 sign flip (Rademacher matrix D)
        x.mul_(self.sign)

        # 3. reshape into 32 768 × 32 768 and run two orthonormal Hadamards
        mat = x.view(BLOCK, BLOCK)                                    # (rows, cols)
        mat = hadamard_transform(mat, scale=1 / math.sqrt(BLOCK))     # along last dim
        mat = hadamard_transform(mat.t().contiguous(), scale=1 / math.sqrt(BLOCK)) # along new last
        x   = mat.t().contiguous().view(PAD_TO)

        # 4. drop padding, uniform row-sample to k dims, √(n/k) scale
        x = x[:n_orig]
        return x[self.idxs] * math.sqrt(n_orig / self.k)