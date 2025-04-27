#!/usr/bin/env python3
"""
Create two JSONL files of Granite embeddings (utterance & context)
that match exactly the structure, key‐order, and float formatting
of your precomputed BERT JSONL.

Usage:
    python make_granite_embeddings.py
"""

import json
import copy
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

# ─── Config ──────────────────────────────────────────────────────────────────
# Path to one existing BERT embeddings line (any line will do)
bert_template_path = Path("bert-output.jsonl")

# Path to your MUStARD‐style dataset
dataset_path       = Path("sarcasm_data.json")

# Output files
utt_out_path       = Path("granite_utterance_embeddings.jsonl")
ctx_out_path       = Path("granite_context_embeddings.jsonl")

# Sentence‐Transformer model
model = SentenceTransformer("ibm-granite/granite-embedding-278m-multilingual")


# ─── Helpers ─────────────────────────────────────────────────────────────────
def to_json_friendly(vec: np.ndarray, decimals: int = 6):
    """
    Convert a float32 numpy vector into a list of Python floats
    rounded to `decimals` places. 6 decimals is enough to
    round‐trip any IEEE-754 float32 exactly, and ensures uniform
    textual width.
    """
    vec32 = vec.astype(np.float32)
    return [float(round(float(x), decimals)) for x in vec32]


# ─── Load a template record from your BERT JSONL ────────────────────────────
with bert_template_path.open("r", encoding="utf-8") as f:
    raw = f.readline()
    bert_template = json.loads(raw)

# Strip out the two fields we’ll overwrite, but keep everything else
template_skel = copy.deepcopy(bert_template)
template_skel.pop("linex_index", None)
template_skel["features"][0]["layers"][0]["values"] = []


# ─── Read your MUStARD‐style JSON dataset ────────────────────────────────────
with dataset_path.open("r", encoding="utf-8") as f:
    data = json.load(f)  # keys are ordered in Python 3.7+

# ─── Open output files ───────────────────────────────────────────────────────
with utt_out_path.open("w", encoding="utf-8") as f_utt, \
     ctx_out_path.open("w", encoding="utf-8") as f_ctx:

    for idx, (key, record) in enumerate(tqdm(data.items(), total=len(data), desc="Encoding")):

        # — Utterance embedding —
        utt_text = record["utterance"]
        utt_vec  = model.encode(utt_text)
        utt_vals = to_json_friendly(utt_vec, decimals=6)

        utt_rec = copy.deepcopy(template_skel)
        utt_rec["linex_index"] = idx
        utt_rec["features"][0]["layers"][0]["values"] = utt_vals
        f_utt.write(json.dumps(utt_rec) + "\n")

        # — Context embedding (concatenate all context turns) —
        ctx_text = " ".join(record.get("context", []))
        ctx_vec  = model.encode(ctx_text)
        ctx_vals = to_json_friendly(ctx_vec, decimals=6)

        ctx_rec = copy.deepcopy(template_skel)
        ctx_rec["linex_index"] = idx
        ctx_rec["features"][0]["layers"][0]["values"] = ctx_vals
        f_ctx.write(json.dumps(ctx_rec) + "\n")


print(f"\n✔ Wrote {len(data)} utterance embeddings to {utt_out_path}")
print(f"✔ Wrote {len(data)} context   embeddings to {ctx_out_path}")
