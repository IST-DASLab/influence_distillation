import torch

def mul(params1, params2):
    return tuple([p1 * p2 for p1, p2 in zip(params1, params2)])

def mul_(params1, params2):
    for p1, p2 in zip(params1, params2):
        p1.mul_(p2)

def add(params1, params2):
    return tuple([p1 + p2 for p1, p2 in zip(params1, params2)])

def add_(params1, params2):
    for p1, p2 in zip(params1, params2):
        p1.add_(p2)

def dot(params1, params2):
    return sum([(p1 * p2).sum() for p1, p2 in zip(params1, params2)])

def scaler_prod(scaler, params):
    return tuple([p * scaler for p in params])

def scaler_prod_(scaler, params):
    for p in params:
        p.mul_(scaler)

def detach(params):
    return tuple([p.detach() for p in params])

def normalize(params):
    norm = (sum([(p**2).sum() for p in params])).sqrt()
    return scaler_prod(1/norm, params)

def normalize_(params):
    norm = (sum([(p**2).sum() for p in params])).sqrt()
    scaler_prod_(1/norm, params)

def flatten(params, dtype=None):
    if dtype is None:
        dtype = params[0].dtype
    return torch.cat([p.flatten().to(dtype) for p in params])

def sign(params):
    return tuple([p.sign() for p in params])

def to(params, arg):
    return tuple([p.to(arg) for p in params])

def numel(params):
    return sum([p.numel() for p in params])

def _sum(params):
    return sum([p.sum() for p in params])