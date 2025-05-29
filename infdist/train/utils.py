import torch
from transformers.tokenization_utils_base import BatchEncoding
from ..utils import tuple_utils

def _ensure_device(x, device):
    if device is None:
        return x
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    elif isinstance(x, (torch.Tensor, BatchEncoding)):
        return x.to(device)
    else:
        raise ValueError(f'Unsupported type: {type(x)}')

def fwd_pass(model, batch, output_hidden_states=False):
    batch = _ensure_device(batch, next(model.parameters()).device)
    outputs = model(input_ids=batch['input_ids'], labels=batch['labels'], output_hidden_states=output_hidden_states)
    if output_hidden_states:
        return outputs
    else:
        return outputs.loss

def fwd_pass_no_reduce(model, batch, reduce_tokens=True):
    batch = _ensure_device(batch, next(model.parameters()).device)
    logits = model(input_ids=batch['input_ids']).logits
    labels = batch['labels']

    shift_labels = labels[..., 1:] # (batch_size, seq_len-1)
    shift_logits = logits[..., :-1, :] # (batch_size, seq_len-1, vocab_size)
    
    batch_size, seq_len_minus1, vocab_size = shift_logits.shape

    # Calculate per-token loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.reshape(-1, vocab_size), shift_labels.reshape(-1)) # (batch_size * seq_len_minus1, )
    loss = loss.reshape(batch_size, seq_len_minus1) # (batch_size, seq_len_minus1)
    breakpoint() #TODO: make sure padding is handled correctly

    if reduce_tokens:
        return loss.sum(-1) / (shift_labels != -100).sum(-1) # (batch_size, )
    else:
        return loss, shift_labels != -100

def zero_grad(params):
    for param in params:
        setattr(param, 'grad', None)

def calc_grad(model, params, batch, create_graph=False, loss_fn=None):
    zero_grad(params)

    assert len(batch['labels']) == 1, 'Batch sizes larger than 1 are not supported'
    if (batch['labels'] == -100).all():
        return tuple([torch.zeros_like(param) for param in params])

    with torch.enable_grad():
        loss = fwd_pass(model, batch) if loss_fn is None else loss_fn(model, batch)
        grad = torch.autograd.grad(loss, params, create_graph=create_graph)
    zero_grad(params)
    return grad

def calc_weighted_grad(model, params, batch, weights, detach_from_model_only=False):
    """
    if detach_from_model_only, then the output will be weighted sum of detached gradients.
    so the graph for the weights will exist. this is more memory efficient, but slower,
    as the individual gradients are calculated.
    """
    zero_grad(params)

    with torch.enable_grad():
        loss = fwd_pass_no_reduce(model, batch)

    assert weights.ndim == 1
    if detach_from_model_only:
        wg = tuple([0 for _ in params])
        for i in range(len(loss)):
            zero_grad(params)
            l = loss[i]
            w = weights[i]
            wg = tuple_utils.add(
                wg,
                tuple_utils.scaler_prod(w, torch.autograd.grad(l/len(loss), params, retain_graph=i<len(loss)-1, create_graph=False))
            )
    else:
        zero_grad(params)
        loss = (loss * weights).mean()
        wg = torch.autograd.grad(loss, params, retain_graph=False, create_graph=True)

    zero_grad(params)
    return wg

def calc_hvp(model, params, batch, vectors):
    assert len(vectors) == len(params)
    zero_grad(params)

    assert len(batch['labels']) == 1, 'Batch sizes larger than 1 are not supported'
    if (batch['labels'] == -100).all(): # invalid sample with no labels
        return tuple([torch.zeros_like(param) for param in params])

    with torch.enable_grad():
        loss = fwd_pass(model, batch)
        param_grads = torch.autograd.grad(loss, params, create_graph=True)
        zero_grad(params)

        gvp = 0
        for v, g in zip(vectors, param_grads):
            vd = v.to(g.device)
            gvp += torch.sum(vd * g)
    
        hvp = torch.autograd.grad(gvp, params)
        zero_grad(params)
    return hvp