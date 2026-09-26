import numpy as np

def save_dataset(group, name, data, opts=4):
    if name in group:
        del group[name]
    if not isinstance(data, (np.ndarray, np.generic)):
        data = np.array(data)
    group.create_dataset(name, data=data, compression="gzip", compression_opts=opts)


def mcc_multilabel(y_true, y_p, is_thresholding=False):
    N, C = y_p.shape
    if not is_thresholding:
        y_pred = y_p
        tp = (y_true * y_pred).sum(axis=0)
        tn = ((1 - y_true) * (1 - y_pred)).sum(axis=0)
        fp = ((1 - y_true) * y_pred).sum(axis=0)
        fn = (y_true * (1 - y_pred)).sum(axis=0)
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.zeros(C, dtype=np.float32)
        valid_mask = denominator > 0
        mcc[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        return mcc
    else:
        best_thres = np.zeros(C, dtype=np.float32)
        best_mcc = np.full(C, -1.0, dtype=np.float32)
        for c in range(C):
            y_p_c = y_p[:, c]
            y_true_c = y_true[:, c]
            total_pos = y_true_c.sum()
            total_neg = N - total_pos
            if total_pos == 0 or total_neg == 0:
                best_thres[c] = 0.5
                best_mcc[c] = 0.0
                continue
            idx = np.argsort(y_p_c)[::-1]
            y_p_sorted = y_p_c[idx]
            y_true_sorted = y_true_c[idx]
            tps = np.cumsum(y_true_sorted)
            fps = np.cumsum(1 - y_true_sorted)
            fns = total_pos - tps
            tns = total_neg - fps
            numerator = (tps * tns) - (fps * fns)
            denominator = np.sqrt((tps + fps) * (tps + fns) * (tns + fps) * (tns + fns))
            mccs = np.full(len(denominator), -1.0, dtype=np.float32)
            valid_mask = denominator > 0
            mccs[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
            best_idx = np.argmax(mccs)
            best_mcc[c] = mccs[best_idx]
            best_thres[c] = y_p_sorted[best_idx]
        return best_thres, best_mcc


def take_at_least_one(y_val_prob, threshold):
    y_val_pred = (y_val_prob >= threshold).astype(int)
    row_sums = y_val_pred.sum(axis=1)
    no_label_indices = row_sums == 0
    if np.any(no_label_indices):
        max_indices = y_val_prob[no_label_indices].argmax(axis=1)
        y_val_pred[no_label_indices, max_indices] = 1
    return y_val_pred


class EarlyStopping:
    def __init__(self, patience=3, delta=0.0, mode="max"):
        self.early_stop = False
        self.patience = patience
        self.counter = 0
        self.best_score = np.inf if mode == "min" else -np.inf
        self.mode = mode
        self.delta = delta

    def __call__(self, score):
        if self.mode == "min":
            improved = score < (self.best_score - self.delta)
        elif self.mode == "max":
            improved = score > (self.best_score + self.delta)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(f"[EarlyStopping] {self.counter}/{self.patience}")
        self.early_stop = self.counter >= self.patience
