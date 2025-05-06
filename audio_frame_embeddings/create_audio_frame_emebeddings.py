#!/usr/bin/env python3
"""
Create 10-fps FRAME-LEVEL embeddings for utterance + context clips.

Output (per sample):
    audio_frame_embeddings/<split>/<id>.npz
        ├─ utter:   (T, 772)  float32
        └─ context: (T, 772)  float32
T = ceil(duration / 0.1s)

Folder layout expected:
    audio_embeddings/audios/{utterances_wav, context_wav}/<id>.wav
    data_splits/{train,val,test}.txt
    data_splits/sarcasm_data.json
"""

import json, argparse, pathlib, math
import numpy as np, torch, torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

# ---------------- config ---------------- #
FRAME_HOP = 0.10      # seconds  (⇒ 10 fps)
FRAME_WIN = 0.24      # seconds
SAMPLE_RATE = 16_000
PROC = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
W2V  = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W2V.to(DEVICE)
# ---------------------------------------- #

def load_mono_16k(path):
    w, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        w = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(w)
    return w[0]            # (T,)

def wav2vec_feats(wav):
    with torch.no_grad():
        inp = PROC(wav.cpu().numpy(), sampling_rate=SAMPLE_RATE,
                   return_tensors="pt", padding=True)
        inp = {k: v.to(DEVICE) for k,v in inp.items()}
        h = W2V(**inp).last_hidden_state.squeeze(0)    # (S,768)  S≈T/20ms
    return h.cpu()                                     # back to CPU

def prosody_stats(wav_seg):
    # wav_seg: (Nsamples,)
    pitch = torchaudio.functional.detect_pitch_frequency(
        wav_seg.unsqueeze(0), SAMPLE_RATE)
    energy = wav_seg.unfold(0, int(FRAME_WIN*SAMPLE_RATE/2),
                               int(FRAME_HOP*SAMPLE_RATE/2)) \
                    .pow(2).mean(-1).sqrt()            # rough RMS inside seg
    return torch.tensor([pitch.mean(), pitch.std(),
                         energy.mean(), energy.std()]) # (4,)

def frame_pool(hidden, tot_len):
    """hidden: (S,768) at 20ms stride → pool to 100ms"""
    h_stride = 0.02                          # 20 ms
    frames = []
    n_frames = math.ceil(tot_len / FRAME_HOP)
    for i in range(n_frames):
        start = int((i*FRAME_HOP)   / h_stride)
        end   = int(((i*FRAME_HOP)+FRAME_WIN) / h_stride)
        seg = hidden[start:end]
        frames.append(seg.mean(0))
    return torch.stack(frames)               # (T,768)

def clip_to_sequence(wav):
    hidden = wav2vec_feats(wav)              # (S,768)
    pooled = frame_pool(hidden, len(wav)/SAMPLE_RATE)  # (T,768)

    # prosody per frame from raw waveform
    step = int(FRAME_HOP * SAMPLE_RATE)
    win  = int(FRAME_WIN * SAMPLE_RATE)
    pros = []
    for i in range(pooled.size(0)):
        beg = i*step
        seg = wav[beg:beg+win]
        if seg.numel() < win:
            seg = torch.nn.functional.pad(seg, (0, win-seg.numel()))
        pros.append(prosody_stats(seg))
    pros = torch.stack(pros)                 # (T,4)

    return torch.cat([pooled, pros], dim=1)  # (T,772)

def process_sample(utt_path, ctx_path):
    utt_wav = load_mono_16k(utt_path)
    ctx_wav = load_mono_16k(ctx_path)
    return clip_to_sequence(utt_wav), clip_to_sequence(ctx_wav)

# ---------------- main ------------------ #

def main(root):
    root = pathlib.Path(root)
    aud_root = root/'audio_embeddings'/'audios'
    out_root = root/'audio_frame_embeddings'
    out_root.mkdir(exist_ok=True)

    split_dir = root/'data_splits'
    with open(split_dir/'sarcasm_data.json') as f:
        meta = json.load(f)

    for split in ["train","val","test"]:
        ids = [ln.strip() for ln in open(split_dir/f"{split}.txt")]
        (out_root/split).mkdir(exist_ok=True)
        for sid in tqdm(ids, desc=f"[{split}]"):
            dest = out_root/split/f"{sid}.npz"
            if dest.exists(): continue
            try:
                utt_f = aud_root/'utterances_wav'/f"{sid}.wav"
                ctx_f = aud_root/'context_wav'/f"{sid}_c.wav"
                utter_seq, ctx_seq = process_sample(utt_f, ctx_f)
                np.savez_compressed(dest,
                                    utter=utter_seq.numpy().astype(np.float32),
                                    context=ctx_seq.numpy().astype(np.float32),
                                    label=np.int8(meta[sid]["sarcasm"]))
            except FileNotFoundError as e:
                print("[warn]", e)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project root")
    main(ap.parse_args().root)

