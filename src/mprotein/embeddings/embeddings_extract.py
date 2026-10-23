import torch


def embeddings_extract(sequence, model=None, tokenizer=None, device="cpu"):
    if torch.cuda.is_available():
        device = "cuda"
    if model == None:
        from esm.models.esmc import EsmcForMaskedLM
        model = EsmcForMaskedLM.from_pretrained("biohub/ESMC-300M").to(device).eval()
    if tokenizer == None:
        from esm.models.esmc import EsmcTokenizer
        tokenizer = EsmcTokenizer()

    X_tokenized = tokenizer(
        sequence,
        padding=True,
        return_tensors="pt",
    )
    # attention_mask = X_tokenized["attention_mask"]
    X_tokenized = X_tokenized.to(device)
    with torch.inference_mode():
        output = model(**X_tokenized, output_hidden_states=True)
    X_embeddings = output.hidden_states[-1].squeeze(0).cpu()
    # attention_mask = attention_mask.cpu()
    return X_embeddings  # , attention_mask
