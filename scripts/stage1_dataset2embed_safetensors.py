import os
import yaml
import torch
import tqdm
import pandas as pd
from transformers import AutoModelForMaskedLM
from safetensors.torch import save_file
from mprotein.embeddings.embeddings_extract import embeddings_extract

with open("configs/config_v1.yaml") as f:
    file = yaml.full_load(f)
    CSV_PATH = file["csv_path"]
    DATA_PATH = file["data_path"]

MODEL_ID = "Synthyra/ESMplusplus_large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PRETRAINED = (
    (AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True))
    .to(DEVICE)
    .eval()
)
TOKENIZER = MODEL_PRETRAINED.tokenizer
Y_LABELS = ["Peripheral", "Transmembrane", "LipidAnchor", "Soluble"]

df_raw = pd.read_csv(CSV_PATH)
df = df_raw.copy()

SAVE_PATH_EMBEDDINGS_SAFETENSORS = os.path.join(DATA_PATH, f"embeddings.safetensors")
SAVE_PATH_TARGETS_SAFETENSORS = os.path.join(DATA_PATH, f"targets.safetensors")

df_partACC = df["PartACC"].values.tolist()
df_Y = df[Y_LABELS].values
df_sequence = df["Sequence"].values.tolist()

tmp_embeddings = {}
tmp_targets = {}

for i, PartACC in tqdm.tqdm(enumerate(df_partACC), total=len(df_partACC)):
    sequence = df_sequence[i]
    X_embedding, _ = embeddings_extract(MODEL_PRETRAINED, TOKENIZER, sequence, DEVICE)

    X_embedding = X_embedding.detach()
    y_target = torch.FloatTensor(df_Y[i]).detach()

    tmp_embeddings[PartACC] = X_embedding
    tmp_targets[PartACC] = y_target

save_file(tmp_embeddings, SAVE_PATH_EMBEDDINGS_SAFETENSORS)
save_file(tmp_targets, SAVE_PATH_TARGETS_SAFETENSORS)
