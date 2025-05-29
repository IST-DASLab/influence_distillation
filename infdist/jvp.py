import torch
torch.set_printoptions(sci_mode=False)
import re


from copy import deepcopy
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual
from infdist.utils.proj import _cosntruct_projector, _project_grad
from tqdm import tqdm


def _get_params(model, param_regex, skip_embd, long_num_params=False):
    if param_regex is not None:
        param_regex = re.compile(param_regex)

    params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if skip_embd and ('embed' in n or 'head' in n):
            continue
        if param_regex is None or param_regex.search(n) is not None:
            params.append(p)
    if long_num_params:
        print(f'Found {len(params)} params.')
    return tuple(params)

@torch.no_grad()
def create_jvp_store(model, loader, num_blocks=4, num_tangents=2, proj_dim=4096, proj_num_parts=1, param_regex=None, skip_embd=False, seed=43):
    device = next(model.parameters()).device

    model.cpu()
    org_model = model
    model = deepcopy(org_model)
    model = model.to(device)

    model.model.layers = model.model.layers[:num_blocks]
    model.lm_head = torch.nn.Identity()

    params = _get_params(model, param_regex, skip_embd)
    org_params = tuple([param.clone() for param in params])

    torch.manual_seed(seed)
    assert proj_num_parts == 1, "proj_num_parts > 1 not supported yet"

    projector = _cosntruct_projector(
        full_dim=num_tangents*model.model.layers[-1].mlp.down_proj.out_features//proj_num_parts,
        proj_dim=proj_dim//proj_num_parts,
        seed=seed,
        device=device,
        dtype=torch.float32
    )

    def _replace_params(new_params):
        with torch.no_grad():
            for param, new_param in zip(params, new_params):
                param.copy_(new_param)

    def _jvp(batch, vec):
        with dual_level():
            dual_params = [make_dual(p, v) for p, v in zip(org_params, vec)]
            _replace_params(dual_params)
            output_dual = model(**{k: v for k, v in batch.items() if k != 'labels'})
            output_tangent = unpack_dual(output_dual.logits.squeeze()[-1]).tangent.squeeze()
            return output_tangent
    
    print("Creating JVP store")
    tangents = [[] for _ in range(len(loader.dataset))]
    for i in range(num_tangents):
        print(f"Creating tangent {i+1}/{num_tangents}")
        vec = tuple([torch.randn_like(param) for param in params])
        for j, sample_j in tqdm(enumerate(loader.dataset), total=len(loader.dataset)):
            sample_j = {k: v.to(device).unsqueeze(0) for k, v in sample_j.items()}
            tangent = _jvp(sample_j, vec)
            tangents[j].append(tangent.cpu())

    print("Projecting")
    embds = []
    for sample_tangents in tqdm(tangents):
        embd = _project_grad(projector, tuple([st.to(device) for st in sample_tangents]), num_parts=1)
        embds.append(embd.detach())

    del model, params, org_params, vec, tangents, sample_tangents
    org_model.to(device)

    return embds