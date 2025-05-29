import torch

def fixed_cross_entropy(source, target, weights, num_items_in_batch=None, ignore_index=-100):
    # batch and seq_len dimensions are merged here

    padding_mask = target == ignore_index
    loss_no_reduce = torch.nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction='none')
    loss_no_reduce.masked_fill_(padding_mask, 0.0)

    # we have a 2d loss here, and we need to weight it
    loss_no_reduce = loss_no_reduce * weights

    loss_sum = loss_no_reduce.sum()
    if num_items_in_batch is None:
        loss = loss_sum / (padding_mask.numel() - padding_mask.sum())
    else:
        loss = loss_sum / num_items_in_batch
    return loss

def calc_loss_fn(model_output, labels, weights, num_items_in_batch: int = None, ignore_index: int = -100):
    logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
    vocab_size = logits.shape[-1]
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    weights = weights.unsqueeze(1).repeat(1, labels.shape[1] - 1).flatten()
    loss = fixed_cross_entropy(logits, shift_labels, weights, num_items_in_batch, ignore_index)
    return loss