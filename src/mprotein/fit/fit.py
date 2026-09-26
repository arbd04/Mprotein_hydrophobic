from mprotein.losses import weighted_focal_loss
import torch
import numpy as np


def model_train(
    MODEL, DEVICE, OPTIMIZER, DATALOADER_TRAIN, CLASS_WEIGHT, CLASS_ALPHA, GAMMA
):
    MODEL.train()
    train_loss = 0.0
    train_counts = 0
    for x, y, mask in DATALOADER_TRAIN:
        OPTIMIZER.zero_grad()
        x, y, mask = (
            x.to(DEVICE, non_blocking=True),
            y.to(DEVICE, non_blocking=True),
            mask.to(DEVICE, non_blocking=True),
        )
        output = MODEL(x, mask)
        loss = weighted_focal_loss(output, y, CLASS_WEIGHT, CLASS_ALPHA, gamma=GAMMA)
        train_loss += loss.item() * y.size(0)
        train_counts += y.size(0)
        loss.backward()
        OPTIMIZER.step()
    return train_loss / train_counts


def model_eval(
    MODEL, DEVICE, DATALOADER_VAL, CLASS_WEIGHT, CLASS_ALPHA, GAMMA, is_loss=True
):
    MODEL.eval()
    eval_loss = 0.0
    eval_counts = 0
    y_eval_prob = []
    y_eval_true = []
    with torch.no_grad():
        for x, y, mask in DATALOADER_VAL:
            x, y, mask = (
                x.to(DEVICE, non_blocking=True),
                y.to(DEVICE, non_blocking=True),
                mask.to(DEVICE, non_blocking=True),
            )
            output = MODEL(x, mask)
            if is_loss:
                loss = weighted_focal_loss(
                    output, y, CLASS_WEIGHT, CLASS_ALPHA, gamma=GAMMA
                )
                eval_loss += loss.item() * y.size(0)
                eval_counts += y.size(0)
            y_eval_prob.append(torch.sigmoid(output).detach().cpu().numpy())
            y_eval_true.append(y.detach().cpu().numpy().astype(int))
    y_eval_prob = np.vstack(y_eval_prob)
    y_eval_true = np.vstack(y_eval_true)
    eval_loss_mean = eval_loss / eval_counts if eval_counts > 0 else 0.0
    return y_eval_prob, y_eval_true, eval_loss_mean
