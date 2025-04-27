#!/usr/bin/env python3
"""
create_audio_embeddings.py

Extracts  [context-wav2vec2 | utterance-wav2vec2 | prosody(4)]
→ 1540-D embeddings and saves one .npy per sample in

    audio_embeddings/train_embeddings/
    audio_embeddings/val_embeddings/
    audio_embeddings/test_embeddings/

according to ID lists in   ../data_splits/{train,val,test}.txt
"""
import os, json, pathlib, argparse
import numpy as np
import torch, torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# ─────────────────────────── directories ──────────────────────────── #

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent         # …/audio_embeddings
PROJECT_DIR = SCRIPT_DIR.parent                              # …/CSCI-535-Project
AUDIO_ROOT  = SCRIPT_DIR                                     # holds audios/  +  *embeddings/
SPLIT_DIR   = PROJECT_DIR / "data_splits"                    # holds json & txt lists

# ────────────────────────── feature helpers ───────────────────────── #

def load_mono_16k(path, target_sr=16000):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav[0]                                            # (T,)

def rms_energy(wav, win=400, hop=160):
    return wav.unfold(0, win, hop).pow(2).mean(-1).sqrt()

@torch.no_grad()
def wav2vec_mean(wav, processor, model, device):
    inp = processor(wav.cpu().numpy(), sampling_rate=16000,
                    return_tensors="pt", padding=True)
    inp = {k: v.to(device) for k, v in inp.items()}
    return model(**inp).last_hidden_state.mean(1).squeeze(0).cpu()   # (768,)

def prosody_vec(wav):
    pitch  = torchaudio.functional.detect_pitch_frequency(wav.unsqueeze(0), 16000)
    energy = rms_energy(wav)
    return torch.tensor([pitch.mean(), pitch.std(), energy.mean(), energy.std()])  # (4,)

# ───────────────────────────── main loop ──────────────────────────── #

def main(root_dir):
    root_dir = pathlib.Path(root_dir)         # normally …/audio_embeddings
    aud_dir  = root_dir / "audios"

    out_dirs = {
        "train": root_dir / "train_embeddings",
        "val":   root_dir / "val_embeddings",
        "test":  root_dir / "test_embeddings"
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ----- metadata & split lists -----------------------------------
    with open(SPLIT_DIR / "sarcasm_data.json") as f:
        meta = json.load(f)

    id2split = {}
    for split in out_dirs:
        with open(SPLIT_DIR / f"{split}.txt") as f:
            id2split.update({line.strip(): split for line in f})

    # ----- models ----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc   = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    w2v    = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device).eval()

    # ----- iterate ---------------------------------------------------
    for i, sample_id in enumerate(meta, 1):
        split = id2split.get(sample_id)
        if split is None:
            continue                     # not in train/val/test lists

        out_path = out_dirs[split] / f"{sample_id}.npy"
        if out_path.exists():
            continue                     # already processed

        try:
            utt = load_mono_16k(aud_dir/"utterances_wav"/f"{sample_id}.wav")
            ctx = load_mono_16k(aud_dir/"context_wav"/f"{sample_id}_c.wav")
        except FileNotFoundError as e:
            print(f"[warn] {e}")
            continue

        vec = torch.cat([wav2vec_mean(ctx, proc, w2v, device),
                         wav2vec_mean(utt, proc, w2v, device),
                         prosody_vec(utt)]).numpy().astype(np.float32)  # (1540,)
        np.save(out_path, vec)

        if i % 100 == 0:
            print(f"{i:6d} clips done …")

    print("✅  Finished — embeddings are in train/val/test folders.")

# ──────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dump 1540-D audio embeddings.")
    ap.add_argument("--root", default=str(AUDIO_ROOT),
                    help="audio_embeddings folder (default: script location)")
    main(ap.parse_args().root)
