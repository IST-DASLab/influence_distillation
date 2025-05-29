import torch
from typing import Union, Optional, Callable
from tqdm import tqdm
import random
import numpy as np
import re

from infdist.utils.proj import _cosntruct_projector, _project_grad
from infdist.utils import tuple_utils
from infdist.utils.rbf import _rbf_kernel, _median_heuristic
from infdist.train.utils import calc_grad


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _log(msg: str):
    print(msg)


def _read_lr_from_optim(
    optimizer: torch.optim.Optimizer
):
    assert optimizer is not None
    lr = optimizer.param_groups[0]['lr']
    if lr == 0:
        # this means warmup was done with a scheduler
        # which has set lr=0 at the end of training
        # we use the average lr
        lr = optimizer.param_groups[0]['initial_lr'] / 2
    return lr


def _read_optim(
    optimizer: torch.optim.Optimizer,
    params: tuple,
):
    if optimizer is None:
        a = tuple([1 for _ in params])
        b = tuple([0 for _ in params])
        return 0, a, b

    
    lr = _read_lr_from_optim(optimizer=optimizer)
    weight_decay = optimizer.param_groups[0].get('weight_decay', 0)

    first_param = optimizer.param_groups[0]['params'][0]
    state = optimizer.state.get(first_param, {})
    step = state.get('step', 0)
    
    _log(f'{step=}, {lr=}, {weight_decay=}')

    if isinstance(optimizer, torch.optim.SGD):
        mu = optimizer.param_groups[0].get('momentum', None)
        m = tuple([optimizer.state.get(param, {}).get('momentum_buffer', None) for param in params])
        assert None not in m, "either the optimizer's pointer to the model is invalid, or there is no state for a param."

        with torch.no_grad():
            a = tuple([1 for _ in params])
            b = tuple_utils.add(
                tuple_utils.scaler_prod(mu, m),
                tuple_utils.scaler_prod(weight_decay, params)
            )
        del m

    elif isinstance(optimizer, (torch.optim.AdamW, torch.optim.Adam)):
        if isinstance(optimizer, torch.optim.Adam):
            assert weight_decay in [None, 0], "Our Adam weighting does not support weight decay. Use AdamW."

        beta1, beta2 = optimizer.param_groups[0].get('betas', (None, None))
        eps = optimizer.param_groups[0].get('eps', 1e-8)
        m = tuple([optimizer.state.get(param, {}).get('exp_avg', None) for param in params])
        v = tuple([optimizer.state.get(param, {}).get('exp_avg_sq', None) for param in params])
        assert None not in m and None not in m, "either the optimizer's pointer to the model is invalid, or there is no state for a param."

        v = tuple_utils.scaler_prod(1 / (1 - beta2 ** (step+1)), v)

        alpha1 = beta1 / (1 - beta1 ** (step+1))
        alpha2 = (1 - beta1) / (1 - beta1 ** (step+1))
        
        with torch.no_grad():
            denom_inv = tuple([1 / (v_i.sqrt() + eps) for v_i in v])
            a = tuple_utils.scaler_prod(alpha2, denom_inv)
            b = tuple_utils.add(
                tuple_utils.scaler_prod(weight_decay, params),
                tuple_utils.scaler_prod(alpha1, tuple_utils.mul(m, denom_inv)),
            )
        del m, v

    else:
        raise NotImplementedError(f'Optimizer {type(optimizer)} is not supported.')

    return lr, a, b


def _optim_to_device(optimizer, device):
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def _get_params(
    model: torch.nn.Module,
    skip_embd: bool = False,
    param_subset: str = None,
    param_regex: str = None,
):
    if param_regex is not None:
        param_regex = re.compile(param_regex)
        
    all_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if skip_embd and ('embed' in n or 'head' in n):
            continue
        if param_regex is not None and not param_regex.search(n):
            continue
        if param_subset is None or param_subset.startswith('random-') or param_subset in n:
            all_params.append(p)
    all_params = tuple(all_params)

    if param_subset is not None and param_subset.startswith('random-'):
        num_params = int(param_subset.split('-')[1])
        if num_params > len(all_params):
            raise ValueError(f"Requested {num_params} params, but only {len(all_params)} available.")
        params_idx = np.random.choice(len(all_params), num_params, replace=False).tolist()
        params = tuple(all_params[i] for i in params_idx)
    else:
        params = all_params

    _log(f"Selected {len(params)} parameters for {skip_embd=} and {param_subset=}")
    return tuple(params)


def _create_vanilla_loader_with_dataset(loader, dataset):
    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=loader.collate_fn,
        batch_size=1,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=None,
        persistent_workers=False,
        timeout=loader.timeout,
        shuffle=False,
    )

def _landmark_p_to_full_krr(
    landmark_idx: torch.Tensor, # (l, )
    embeddings: torch.Tensor, # (n, e)
    p_L: torch.Tensor, # (l, )
    damp: float,
    sigma: float = None,
    chunk_size: int = 1024,
):
    L = embeddings[landmark_idx].float() # (l, e)
    if sigma is None:
        sigma = _median_heuristic(L)

    K = _rbf_kernel(L, L, sigma, chunk=chunk_size)  # (l, l)
    K.diagonal().add_(damp * K.diag().mean())
    v = torch.linalg.solve(K, p_L.float())
    
    K_E = _rbf_kernel(embeddings.float(), L, sigma, chunk=chunk_size)  # (N, l)

    p = K_E @ v
    return p.to(p_L.dtype)

def _perform_selection_rr(
    all_p_i_numpy: list,
    k: int,
    device: Optional[torch.device] = None,
):
    selected_indices_list = []
    selected_weights = []
    num_sampled = 0

    # Work on copies to avoid modifying the original all_p_i_numpy
    working_scores = [np.copy(p_arr) for p_arr in all_p_i_numpy]
    num_targets_for_selection = len(working_scores)

    # Loop until k indices are selected or no more valid items can be found
    while num_sampled < k and num_targets_for_selection > 0:
        made_selection_in_round = False
        for target_idx in np.random.permutation(num_targets_for_selection): # shuffle for fairness
            if num_sampled >= k:
                break # k items already selected

            # Find index of max score for current target. Assumes working_scores[target_idx] is not empty.
            current_max_idx = np.argmax(working_scores[target_idx])
            current_max_score = working_scores[target_idx][current_max_idx]

            # If the max score is not our "already selected" marker value (-1e8)
            if current_max_score > -1e8: # Using strict >. Assumes -1e8 is the exact marker.
                selected_indices_list.append(current_max_idx)
                selected_weights.append(current_max_score)
                num_sampled += 1
                made_selection_in_round = True

                # Mark this selected item (index) as "used" across all targets
                # by setting its score to the low marker value.
                for scores_for_one_target in working_scores:
                    scores_for_one_target[current_max_idx] = -1e8
            
            if num_sampled >= k: # Check again if k was reached by this selection
                break 

        if num_sampled >= k: # Ensure exit from while loop if k reached
            break 

        # If a full round yielded no new selections, stop to prevent infinite loop.
        if not made_selection_in_round:
            break

    # Convert the list of selected indices to a PyTorch tensor.
    selected_idx = torch.tensor(selected_indices_list, dtype=torch.long, device=device)
    selected_weights = torch.tensor(selected_weights, dtype=torch.float32, device=device)
    return selected_idx, selected_weights

    

@torch.no_grad()
def _mask(tensors, masks):
    if masks is None:
        return tensors
    return tuple([t[m].flatten() for t, m in zip(tensors, masks) if m.any()])


@torch.no_grad()
def _create_grad_store(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    params: tuple,
    optim_a: tuple,
    projector: torch.nn.Module,
    masks: tuple,
    proj_num_parts: int,
    out_device: Union[str, torch.device] = None,
    normalize: bool = True,
    loss_fn: Callable = None,
):
    if out_device is None:
        out_device = next(model.parameters()).device

    if optim_a is not None:
        optim_a = _mask(optim_a, masks)

    store = []
    for j, sample_j in tqdm(enumerate(loader), total=len(loader)):
        gj = calc_grad(model, params, sample_j, loss_fn=loss_fn)
        gj = _mask(gj, masks)

        if optim_a is not None:
            tuple_utils.mul_(gj, optim_a)
        if normalize:
            tuple_utils.normalize_(gj)
        store.append(_project_grad(
            projector=projector,
            grads_tuple=gj,
            num_parts=proj_num_parts
        ).to(out_device))
    return torch.stack(store)


def _create_masks(params, mask_numel):
    if mask_numel is not None:
        mask_numels = [round(mask_numel * param.numel() / tuple_utils.numel(params)) for param in params]
        masks = []
        _log('Generating masks...')
        for n, param in zip(mask_numels, params):
            if n >= param.numel():
                masks.append(torch.ones_like(param, dtype=torch.bool))
                continue
            # Generate random floats and take top-n indices
            numel = param.numel()
            rand_vals = torch.rand(numel, device=param.device)
            topk = torch.topk(rand_vals, n, largest=True).indices

            mask = torch.zeros(numel, dtype=torch.bool, device=param.device)
            mask[topk] = True

            masks.append(mask.view(param.shape))
            
        masks = tuple(masks)

        _log(f'Generated masks with numel: {tuple_utils._sum(masks).item()}')
    else:
        masks = None
    return masks


@torch.no_grad()
def pick_first_order(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    target_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    embeddings: torch.Tensor,
    k: int,
    skip_embd: bool = False,
    proj_type: str = 'had',
    proj_dim: int = 32768,
    proj_num_parts: int = 1,
    proj_subset: str = 'down_proj', # 'random-{}' or a key like 'o_proj', or None
    param_regex: str = None,
    param_mask_numel: int = None,
    num_landmarks: int = 4096,
    seed: int = 1,
    damp: float = 1e-2,
    return_intermediate=False,
    landmark_idx=None,
    loss_fn: Callable = None,
):
    _set_seed(seed)
    device = next(model.parameters()).device
    
    params = _get_params(
        model,
        skip_embd=skip_embd,
        param_subset=proj_subset,
        param_regex=param_regex
    )

    # handle the model/optimizer device to avoid OOM
    model.cpu()
    _optim_to_device(optimizer=optimizer, device='cpu')
    lr, a, b = _read_optim(optimizer=optimizer, params=params)
    del lr, b
    a = tuple_utils.to(a, device) if optimizer is not None else a
    model.to(device)

    # make sure model is in eval mode
    model_training = model.training
    model.eval()

    masks = _create_masks(
        params=params,
        mask_numel=param_mask_numel
    )

    # create the projector
    projector = _cosntruct_projector(
        full_dim=tuple_utils.numel(params) if masks is None else tuple_utils._sum(masks).item(),
        proj_dim=proj_dim//proj_num_parts,
        seed=seed,
        device=device,
        dtype=torch.float32 if proj_type == 'rad' else params[0].dtype,
        proj_type=proj_type
    )

    # def generate landmarks
    if landmark_idx is None:
        landmark_idx = np.random.choice(
            len(embeddings),
            size=num_landmarks,
            replace=False
        )
    
    # create the orojected gradient stores
    landmark_store = _create_grad_store(
        model=model,
        loader=_create_vanilla_loader_with_dataset(train_loader, train_loader.dataset.select(landmark_idx)),
        params=params,
        optim_a=a if optimizer is not None else None,
        projector=projector,
        masks=masks,
        proj_num_parts=proj_num_parts,
        out_device=device,
        normalize=True,
        loss_fn=loss_fn
    ) # (l, p)

    target_store = _create_grad_store(
        model=model,
        loader=_create_vanilla_loader_with_dataset(target_loader, target_loader.dataset),
        params=params,
        optim_a=None, # no optim_a for target
        projector=projector,
        masks=masks,
        proj_num_parts=proj_num_parts,
        out_device=device,
        normalize=True,
        loss_fn=loss_fn
    ) # (t, p)

    # calculate p for landmarks
    p_L = torch.mm(
        landmark_store,
        target_store.mT
    ) # (l, t)

    # convert to full p and select
    num_targets = target_store.shape[0]
    all_p_i_numpy = []
    for i in range(num_targets):
        p_i = _landmark_p_to_full_krr(
            landmark_idx=torch.as_tensor(landmark_idx, device=device, dtype=torch.long),
            embeddings=embeddings,
            p_L=p_L[:, i],
            damp=damp
        )
        all_p_i_numpy.append(p_i.detach().float().cpu().numpy())

    selected_idx, weights = _perform_selection_rr(
        all_p_i_numpy=all_p_i_numpy,
        k=k,
        device=device
    )

    # back to original device and training 
    _optim_to_device(optimizer=optimizer, device=device)
    model.train(model_training)

    if return_intermediate:
        return selected_idx.cpu().numpy(), weights.cpu().numpy(), landmark_idx, landmark_store, target_store, p_L
    else:
        return selected_idx.cpu().numpy()