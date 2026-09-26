import os

# import requests
import numpy as np

# import pandas as pd

from scipy.spatial.distance import pdist, squareform
from Bio.PDB import PDBList, PDBParser, is_aa

D3T1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def extract_contact_map(pdb_id, chain_id, threshold=8.0, pdb_save_path="./data/pdb"):
    os.makedirs(pdb_save_path, exist_ok=True)
    pdb_file = os.path.join(pdb_save_path, f"pdb{pdb_id.lower()}.ent")

    if not os.path.exists(pdb_file):
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_save_path, file_format="pdb")

    if not os.path.exists(pdb_file):
        return None, None

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]
    chain = model[chain_id]

    coords = []
    amino_acids = []

    for res in chain:
        res_name = res.resname
        # 표준 아미노산이고 좌표가 존재하는지 확인
        if is_aa(res) and res_name in D3T1:
            try:
                # CB(C-beta)가 있으면 CB, 없으면(Glycine) CA 좌표 추출
                if "CB" in res:
                    coords.append(res["CB"].get_coord())
                elif "CA" in res:
                    coords.append(res["CA"].get_coord())
                else:
                    continue
                amino_acids.append(D3T1[res_name])
            except:
                continue
    if not coords:
        return None, None

    # 1:1 매칭을 위한 최종 서열 생성
    sequence = "".join(amino_acids)
    coords = np.array(coords)  # (L, 3)
    dist_matrix = squareform(pdist(coords, metric="euclidean"))
    contact_map = (dist_matrix < threshold).astype(int)

    return contact_map, sequence


# df_raw = pd.read_csv(
#     r"C:\Users\khj04\github\Mprotein_hydrophobic\data\raw\dataset_from_deeploc2_1.csv"
# )
# df = df_raw.copy()

# acc_uniprot = ",".join(df["ACC"].tolist())

# with open("ids.txt", "w", encoding="utf-8") as f:
#     f.write(acc_uniprot)

df_raw = pd.read_csv(r"C:\Users\khj04\github\Mprotein_hydrophobic\data\raw\ACC_PDB.csv")
df = df_raw.copy()
pdbidlist = df["PDB"].tolist()
output_file = "./output.csv"
query = """query($my_ids: [String!]!) {
    entries(entry_ids: $my_ids) {
        rcsb_id
        exptl { method }
        rcsb_entry_info { resolution_combined }
        polymer_entities {
            rcsb_polymer_entity_container_identifiers {
                uniprot_ids
                auth_asym_ids
                rcsb_id
            }
            entity_poly { pdbx_seq_one_letter_code_can }
        }
    }
}
"""
full = {"entries": []}
batch_size = 100
for i in range(0, len(pdbidlist), batch_size):
    batch = pdbidlist[i : i + batch_size]
    response = requests.post(
        "https://data.rcsb.org/graphql",
        json={"query": query, "variables": {"my_ids": batch}},
    )
    if response.status_code == 200:
        batch_data = response.json().get("data", {}).get("entries", [])

        rows = []
        for entry in batch_data:
            pdb_id = entry.get("rcsb_id")
            method = entry.get("exptl", [{}])[0].get("method", "N/A")
            # 해상도는 리스트 형태이므로 첫 번째 값 추출
            res_list = entry.get("rcsb_entry_info", {}).get("resolution_combined", [])
            resolution = res_list[0] if res_list else None

            # 각 엔티티(단백질 부품)를 순회
            for entity in entry.get("polymer_entities", []):

                # UniProt ID와 체인 기호(auth_asym_ids) 추출
                identifiers = entity.get(
                    "rcsb_polymer_entity_container_identifiers", {}
                )
                uniprot_ids = ";".join(identifiers.get("uniprot_ids") or [])

                # 나중에 PDB 좌표를 열 때 쓸 '대표 체인' (첫 번째 체인)
                auth_chains = identifiers.get("auth_asym_ids", [])
                target_chain = auth_chains[0] if auth_chains else "N/A"

                entity_id = identifiers.get("rcsb_id")

                # ESM-2 입력용 표준 서열
                sequence = entity.get("entity_poly", {}).get(
                    "pdbx_seq_one_letter_code_can", ""
                )
                seq_len = len(sequence)

                # 데이터 행 구성
                rows.append(
                    {
                        "PDB_ID": pdb_id,
                        "Entity_ID": entity_id,
                        "Target_Chain": target_chain,
                        "UniProt_ID": uniprot_ids,
                        "Resolution": resolution,
                        "Method": method,
                        "Sequence": sequence,
                        "Length": seq_len,
                    }
                )
        df_batch = pd.DataFrame(rows)
        if not os.path.isfile(output_file):
            df_batch.to_csv(
                output_file, index=False, encoding="utf-8-sig"
            )  # 처음엔 헤더 포함
        else:
            df_batch.to_csv(
                output_file, index=False, mode="a", header=False, encoding="utf-8-sig"
            )
