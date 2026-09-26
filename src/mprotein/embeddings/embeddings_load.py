import os
import torch

# import h5py
# from torch_geometric.data import Data
from safetensors import safe_open
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def padding_collate_fn(batch):
    embeddings = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    # 미니배치(w. 패딩) & 마스크 생성
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0)
    attention_mask = (padded_embeddings.abs().sum(dim=-1) != 0).float()
    targets = torch.stack(targets)
    return padded_embeddings, targets, attention_mask


class K_CV_Dataset(Dataset):
    def __init__(self, K_CV, validation_k, save_path, is_train=True):
        self.save_path = save_path
        self.path_x = os.path.join(self.save_path, f"embeddings.safetensors")
        self.path_y = os.path.join(self.save_path, f"targets.safetensors")
        self.handle_x = None
        self.handle_y = None
        if is_train:
            self.train_part = [i for i in range(K_CV) if i != validation_k]
        else:
            self.train_part = [validation_k]
        self.train_keys = []
        with safe_open(self.path_y, framework="pt") as f:
            all_keys = f.keys()
            for p in self.train_part:
                prefix = f"part_{p}_"
                p_keys = [k for k in all_keys if k.startswith(prefix)]
                self.train_keys.extend(p_keys)
        self.train_keys.sort()

    def __len__(self):
        return len(self.train_keys)

    def open_files(self):
        # 멀티프로세싱의 각 worker 안에서 파일을 처음 한 번만 엽니다. > 뭔말인지 잘 이해 못함 솔직히 num worker 각각이 핸들 한번씩 연다는말같긴한데..
        if self.handle_x is None:
            self.handle_x = safe_open(self.path_x, framework="pt")
        if self.handle_y is None:
            self.handle_y = safe_open(self.path_y, framework="pt")

    def __del__(self):
        if self.handle_x is not None:
            self.handle_x = None
        if self.handle_y is not None:
            self.handle_y = None

    def __getitem__(self, idx):
        self.open_files()
        key = self.train_keys[idx]
        embedding = self.handle_x.get_tensor(key)
        target = self.handle_y.get_tensor(key)
        return embedding, target


"""
@torch.no_grad()
def topk_edges_from_adj(A, k=32, add_loop=True):
    from torch_geometric.utils import add_self_loops
    L = A.size(0)
    A = A.clone()
    k = min(k, L-1)

    vals, idx = torch.topk(A, k=k, dim=1)
    row = torch.arange(L, device=A.device).unsqueeze(1).expand(L, k).reshape(-1)
    col = idx.reshape(-1)
    w = vals.reshape(-1)

    edge_index = torch.stack([row, col], dim=0)
    edge_weight = w

    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

    if add_loop:
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight=edge_weight, fill_value=1.0, num_nodes=L)

    return edge_index, edge_weight


class K_CV_PyGDataset(Dataset):
    def __init__(self, K_CV, validation_k, k_, save_path, is_train=True):
        self.k_ = k_
        self.save_path = save_path
        self.path_x = os.path.join(self.save_path, "embeddings.safetensors")
        self.path_y = os.path.join(self.save_path, "targets.safetensors")
        self.path_z = os.path.join(self.save_path, "maps.h5")
        self.handle_x = None
        self.handle_y = None
        self.handle_z = None
        if is_train:
            self.train_part = [i for i in range(K_CV) if i != validation_k]
        else:
            self.train_part = [validation_k]
        self.train_keys = []
        with safe_open(self.path_y, framework="pt") as f:
            all_keys = f.keys()
            for p in self.train_part:
                prefix = f"part_{p}_"
                p_keys = [k for k in all_keys if k.startswith(prefix)]
                self.train_keys.extend(p_keys)
        self.train_keys.sort()

    def __len__(self):
        return len(self.train_keys)

    def open_files(self):
        if self.handle_x is None:
            self.handle_x = safe_open(self.path_x, framework="pt")
        if self.handle_y is None:
            self.handle_y = safe_open(self.path_y, framework="pt")
        if self.handle_z is None:
            self.handle_z = h5py.File(self.path_z, "r")

    def __del__(self):
        if self.handle_x is not None:
            self.handle_x = None
        if self.handle_y is not None:
            self.handle_y = None
        if self.handle_z is not None:
            self.handle_z.close()
            self.handle_z = None

    def __getitem__(self, idx):
        self.open_files()
        key = self.train_keys[idx]
        embedding = self.handle_x.get_tensor(key)[1:-1]
        target = self.handle_y.get_tensor(key)
        graph = torch.from_numpy(self.handle_z[key]["contact_map"][:]).float()
        edge_index, edge_weight = topk_edges_from_adj(graph, k=self.k_, add_loop=True)
        data = Data(x=embedding, edge_index=edge_index, edge_weight=edge_weight, y=target)
        return data
"""
