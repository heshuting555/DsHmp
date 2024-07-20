import torch


def multi_pos_cross_entropy(
    pred, aux_pred, label, pos_normalize=True,
):

    valid_mask = label.sum(1) != 0
    pred = pred[valid_mask]
    label = label[valid_mask]
    if min(pred.shape) != 0:
        logits_max, _ = torch.max(pred, dim=1, keepdim=True)
        logits = pred - logits_max.detach()
    else:
        logits = pred

    if pos_normalize:
        pos_norm = torch.div(label, label.sum(1).reshape(-1, 1))
        exp_logits = (torch.exp(logits)) * pos_norm + (
            torch.exp(logits)
        ) * torch.logical_not(label)
    else:
        exp_logits = torch.exp(logits)
    exp_logits_input = exp_logits.sum(1, keepdim=True)
    log_prob = logits - torch.log(exp_logits_input)

    mean_log_prob_pos = (label * log_prob).sum(1) / label.sum(1)
    loss_contrastive = -mean_log_prob_pos

    aux_loss = (torch.abs(aux_pred - label) ** 2).mean()
    c_loss = 2 * loss_contrastive.mean() + aux_loss
    return c_loss


