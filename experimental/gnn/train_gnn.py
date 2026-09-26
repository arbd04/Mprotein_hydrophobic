import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gc
import copy
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from safetensors.torch import save_model
from mprotein.models.stack_ensemble import MetaModel_pred
from src.mprotein.utils import mcc_multilabel, take_at_least_one, EarlyStopping
from experimental.gnn.fitting_gnn import model_train_pyg, model_eval_pyg
from experimental.gnn.model_gnn import Deeploc2_1_PyG, ModelConfigPyG
from mprotein.data.embeddings_load import K_CV_PyGDataset

with open("configs/config_v2.yaml") as f:
    file = yaml.full_load(f)
    weight_path = file["weights_path"]
    DATA_PATH = file["data_path"]
    K_CV = file["k_cv"]
    NUM_EPOCHS = file["num_epochs"]
    IR = file["learning_rate"]
    WD = file["weight_decay"]
    loader_kwargs = file["loader_kwargs"]
    loss = file["loss"]
    model = file["model"]
CFG = ModelConfigPyG(**model)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
y_prob_oof = []
y_true_oof = []
for k in range(K_CV):
    # 1. 초기화
    es = EarlyStopping(patience=3, delta=0.0, mode="max")
    best_mcc = -np.inf
    best_thres = np.zeros(4, dtype=np.float32)
    best_weights = None
    best_epoch = 0
    MODEL = Deeploc2_1_PyG(CFG).to(DEVICE)
    OPTIMIZER = AdamW(MODEL.parameters(), lr=IR, weight_decay=WD)
    DATASET_TRAIN = K_CV_PyGDataset(K_CV, k, 32, DATA_PATH, is_train=True)
    DATASET_TEST = K_CV_PyGDataset(K_CV, k, 32, DATA_PATH, is_train=False)
    DATALOADER_TRAIN = DataLoader(DATASET_TRAIN, shuffle=True, **loader_kwargs)
    DATALOADER_THRES = DataLoader(DATASET_TRAIN, shuffle=False, **loader_kwargs)
    DATALOADER_TEST = DataLoader(DATASET_TEST, shuffle=False, **loader_kwargs)

    # 2. 학습 진행
    epoch_iterator = tqdm(range(NUM_EPOCHS), desc=f"[Fold {k+1}/{K_CV}]")
    for epoch in epoch_iterator:
        epoch_now = epoch + 1
        epoch_iterator.write(f"Epoch {epoch_now}/{NUM_EPOCHS}")
        ## TRAIN
        train_loss_mean = model_train_pyg(
            MODEL, DEVICE, OPTIMIZER, DATALOADER_TRAIN, **loss
        )
        epoch_iterator.write(f"TRAIN done")
        epoch_iterator.write(f"Train Loss: {train_loss_mean}")

        ## THRESHOLD
        y_thres_prob, y_thres_true, _ = model_eval_pyg(
            MODEL,
            DEVICE,
            DATALOADER_THRES,
            **loss,
            is_loss=False,
        )
        threshold, mcc_thres = mcc_multilabel(
            y_thres_true, y_thres_prob, is_thresholding=True
        )
        epoch_iterator.write(f"THRESHOLD done | {threshold}")
        epoch_iterator.write(f"Threshold Macro MCC: {mcc_thres.mean()} ({mcc_thres})")
        del _

        ## VALIDATION
        y_val_prob, y_val_true, val_loss_mean = model_eval_pyg(
            MODEL,
            DEVICE,
            DATALOADER_TEST,
            **loss,
            is_loss=True,
        )
        y_val_pred = take_at_least_one(y_val_prob, y_val_true, threshold)
        mcc_val_per = mcc_multilabel(y_val_true, y_val_pred, is_thresholding=False)
        mcc_val = mcc_val_per.mean()

        if mcc_val > best_mcc:
            best_epoch = epoch_now
            best_mcc = mcc_val
            best_thres = threshold
            best_weights = copy.deepcopy(MODEL.state_dict())
            best_y_val_prob = y_val_prob
            best_y_val_true = y_val_true

        epoch_iterator.write(f"VALIDATION done")
        epoch_iterator.write(f"Validation Loss: {val_loss_mean}")
        epoch_iterator.write(f"Validation Macro MCC: {mcc_val} ({mcc_val_per}))")

        # Early Stop
        es(mcc_val)
        if es.early_stop == True:
            epoch_iterator.write(f"{k} Fold Early Stopped")
            break
        # scheduler.step(mcc_val)

        del (
            train_loss_mean,
            val_loss_mean,
            y_val_prob,
            y_val_true,
            y_val_pred,
            mcc_val_per,
            mcc_val,
            threshold,
            mcc_thres,
            y_thres_prob,
            y_thres_true,
        )
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Fold {k} Best at Epoch {best_epoch}")
    print(f"Best MCC is {best_mcc}")
    print(f"Best Thres is {best_thres}")
    MODEL.load_state_dict(best_weights)
    save_model(MODEL, os.path.join(weight_path, f"weights_fold_{k}.safetensors"))
    y_prob_oof.append(best_y_val_prob)
    y_true_oof.append(best_y_val_true)
    del (
        MODEL,
        DATALOADER_TRAIN,
        DATALOADER_THRES,
        DATALOADER_TEST,
        DATASET_TRAIN,
        DATASET_TEST,
        es,
        best_epoch,
        best_mcc,
        best_thres,
        best_weights,
        best_y_val_prob,
        best_y_val_true,
    )
    gc.collect()
    torch.cuda.empty_cache()
y_prob_oof = np.vstack(y_prob_oof)
y_true_oof = np.vstack(y_true_oof)
threshold_oof, mcc_thres_oof = mcc_multilabel(
    y_true_oof, y_prob_oof, is_thresholding=True
)
print(f"Out of Fold)")
print(f"OOF Threshold is {threshold_oof}")
print(f"OOF Threshold MCC is {mcc_thres_oof}")
y_pred_oof = MetaModel_pred(y_prob_oof, y_true_oof, weight_path)
best_threshold, best_mcc = mcc_multilabel(y_true_oof, y_pred_oof, is_thresholding=True)
print(f"Meta Threshold: {best_threshold}")
print(f"Meta Macro MCC: {best_mcc}")
