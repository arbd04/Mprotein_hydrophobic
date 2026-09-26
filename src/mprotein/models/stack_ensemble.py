import os
import numpy as np
import pickle
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier


def contact_Regression(amap, cmap, penalty_, save_path):
    save_contact = os.path.join(save_path, "contact_model.pickle")
    contact_model = LogisticRegression(
        random_state=42,
        l1_ratio=1,
        solver="saga",
        C=penalty_,
        class_weight="balanced",
        max_iter=1000,
    )
    contact_model.fit(amap, cmap)
    joblib.dump(contact_model, save_contact)
    print("contact model saved.")


def MetaModel_pred(y_prob_oof, y_true_oof, save_path):
    save_Meta = os.path.join(save_path, "meta_model.pickle")
    meta_model = MultiOutputClassifier(LogisticRegression(random_state=42))
    meta_model.fit(y_prob_oof, y_true_oof)
    output = meta_model.predict_proba(y_prob_oof)
    y_pred_oof = np.array([p[:, 1] for p in output]).T
    with open(save_Meta, "wb") as f:
        pickle.dump(meta_model, f)
    return y_pred_oof
