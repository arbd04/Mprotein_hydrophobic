import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mprotein.losses.weighted_focal_loss import weighted_focal_loss
import torch
import numpy as np


def model_train_pyg(
    MODEL, DEVICE, OPTIMIZER, DATALOADER_TRAIN, CLASS_WEIGHT, CLASS_ALPHA, GAMMA
):
    MODEL.train()
    train_loss = 0.0
    train_counts = 0
    for data in DATALOADER_TRAIN:
        OPTIMIZER.zero_grad()
        data = data.to(DEVICE, non_blocking=True)
        y = data.y
        output = MODEL(data)
        loss = weighted_focal_loss(output, y, CLASS_WEIGHT, CLASS_ALPHA, gamma=GAMMA)
        train_loss += loss.item() * data.num_graphs
        train_counts += data.num_graphs
        loss.backward()
        OPTIMIZER.step()
    return train_loss / max(train_counts, 1)


def model_eval_pyg(
    MODEL, DEVICE, DATALOADER_VAL, CLASS_WEIGHT, CLASS_ALPHA, GAMMA, is_loss=True
):
    MODEL.eval()
    eval_loss = 0.0
    eval_counts = 0
    y_eval_prob = []
    y_eval_true = []
    with torch.no_grad():
        for data in DATALOADER_VAL:
            data = data.to(DEVICE, non_blocking=True)
            y = data.y
            output = MODEL(data)
            if is_loss:
                loss = weighted_focal_loss(
                    output, y, CLASS_WEIGHT, CLASS_ALPHA, gamma=GAMMA
                )
                eval_loss += loss.item() * data.num_graphs
                eval_counts += data.num_graphs
            y_eval_prob.append(torch.sigmoid(output).detach().cpu().numpy())
            y_eval_true.append(y.detach().cpu().numpy().astype(int))
    y_eval_prob = np.vstack(y_eval_prob)
    y_eval_true = np.vstack(y_eval_true)
    eval_loss_mean = eval_loss / max(eval_counts, 1)
    return y_eval_prob, y_eval_true, eval_loss_mean
