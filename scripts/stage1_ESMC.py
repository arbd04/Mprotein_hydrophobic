import os
import yaml
import torch
import tqdm
import h5py
import pandas as pd
from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
from mprotein.embeddings.embeddings_extract import embeddings_extract

with open("configs/config_v1.yaml") as f:
    file = yaml.full_load(f)
    DATASETS_PATH = file["datasets_path"]
    EMBEDDINGS_PATH = file["embeddings_path"]

MODEL_ID = "biohub/ESMC-300M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PRETRAINED = EsmcForMaskedLM.from_pretrained(MODEL_ID).to(DEVICE).eval()
TOKENIZER = EsmcTokenizer()

df_raw = pd.read_csv(os.path.join(DATASETS_PATH, "dataset_from_deeploc2_1.csv"))
df = df_raw.copy()


with h5py.File(os.path.join(EMBEDDINGS_PATH, "embeddings_test.h5"), "a") as f:
    for acc, seq in zip(df["ACC"], df["Sequence"]):
        if acc in f:
            continue
        grp_acc = f.create_group(acc)
        grp_acc.create_dataset(
            "embedding",
            data=embeddings_extract(seq, MODEL_PRETRAINED, TOKENIZER, DEVICE),
        )
