import torch


def embeddings_extract(model, tokenizer, sequence, device):
    X_tokenized = tokenizer(
        sequence,
        padding=True,
        return_tensors="pt",
    )
    attention_mask = X_tokenized["attention_mask"]
    X_tokenized = X_tokenized.to(device)
    with torch.no_grad():
        output = model(**X_tokenized, output_hidden_states=True)
    X_embeddings = output.hidden_states[-1].squeeze(0).cpu()
    attention_mask = attention_mask.cpu()
    return X_embeddings, attention_mask
