#!/usr/bin/env python3
"""
create_audio_frame_embeddings.py
--------------------------------
Generate 10-fps FRAME-LEVEL embeddings for every sample:

    context sequence : (Tc , 772)
    utter   sequence : (Tu , 772)

Stored as compressed .npz files under  audio_frame_embeddings/<split>/.

Folder layout expected:
    audio_embeddings/audios/{utterances_wav,context_wav}/<ID>.wav
    data_splits/{train,val,test}.txt
    data_splits/sarcasm_data.json
"""

import argparse, json, math, pathlib, numpy as np, torch, torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

# ─────────────── parameters ──────────────── #
FRAME_HOP = 0.10          # 100 ms → 10 fps
FRAME_WIN = 0.24          # 240 ms analysis window
SR        = 16_000
PROC = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
W2V  = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W2V.to(DEVICE)
# ─────────────────────────────────────────── #

def load_mono_16k(path: pathlib.Path):
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    return wav[0]                             # (T,)

@torch.no_grad()
def wav2vec_hidden(wav: torch.Tensor):
    """hidden states at ~20 ms stride → (S , 768) on CPU"""
    x = PROC(wav.cpu().numpy(), sampling_rate=SR,
             return_tensors="pt", padding=True)
    x = {k: v.to(DEVICE) for k, v in x.items()}
    return W2V(**x).last_hidden_state.squeeze(0).cpu()

def prosody_stats(segment: torch.Tensor):
    """Return 4-d [f0_mean, f0_std, rms_mean, rms_std] for a waveform segment"""
    pitch  = torchaudio.functional.detect_pitch_frequency(segment.unsqueeze(0), SR)
    rms    = segment.pow(2).mean().sqrt()
    return torch.tensor([pitch.mean(), pitch.std(), rms, 0.0])  # rms std not meaningful on 1 seg

def pool_to_10fps(hidden: torch.Tensor, wav_len: float):
    """hidden 20 ms stride → mean-pool into 100 ms frames (WINDOW 240 ms)."""
    h_stride = 0.02                                       # seconds between hidden steps
    n_frames = math.ceil(wav_len / FRAME_HOP)
    span     = int(FRAME_WIN / h_stride)
    hop      = int(FRAME_HOP / h_stride)
    pooled = []
    for i in range(n_frames):
        start = i*hop
        end   = start + span
        seg = hidden[start:end]
        pooled.append(seg.mean(0))
    return torch.stack(pooled)                            # (T , 768)

def sequence_from_clip(wav: torch.Tensor):
    """Return frame sequence (T , 772)"""
    hidden = wav2vec_hidden(wav)                         # (S , 768)
    pooled = pool_to_10fps(hidden, len(wav)/SR)          # (T , 768)

    # prosody per 100 ms frame
    hop_samp = int(FRAME_HOP*SR)
    win_samp = int(FRAME_WIN*SR)
    frames = []
    for i in range(pooled.size(0)):
        beg = i*hop_samp
        seg = wav[beg:beg+win_samp]
        if seg.numel() < win_samp:                       # pad final segment
            seg = torch.nn.functional.pad(seg, (0, win_samp-seg.numel()))
        frames.append(torch.cat([pooled[i], prosody_stats(seg)], dim=0))
    return torch.stack(frames)                           # (T , 772)

def process_pair(utt_path, ctx_path):
    utt = load_mono_16k(utt_path)
    ctx = load_mono_16k(ctx_path)
    return sequence_from_clip(utt), sequence_from_clip(ctx)

# ───────────────────────────── main ───────────────────────────── #

def main(root="."):
    root = pathlib.Path(root)
    aud_root = root/'audio_embeddings'/'audios'
    split_dir = root/'data_splits'
    out_root  = root/'audio_frame_embeddings'
    out_root.mkdir(exist_ok=True)

    with open(split_dir/'sarcasm_data.json') as f:
        meta = json.load(f)

    for split in ("train","val","test"):
        ids = [ln.strip() for ln in open(split_dir/f"{split}.txt")]
        (out_root/split).mkdir(exist_ok=True)

        for sid in tqdm(ids, desc=f"{split:>5}"):
            dest = out_root/split/f"{sid}.npz"
            if dest.exists():
                continue
            try:
                utt_f = aud_root/'utterances_wav'/f"{sid}.wav"
                ctx_f = aud_root/'context_wav'/f"{sid}_c.wav"
                utt_seq, ctx_seq = process_pair(utt_f, ctx_f)
                np.savez_compressed(dest,
                                    utter=utt_seq.numpy().astype(np.float32),
                                    context=ctx_seq.numpy().astype(np.float32),
                                    label=np.int8(meta[sid]["sarcasm"]))
            except FileNotFoundError as e:
                print("[warn]", e)

    print("✅  all splits finished → audio_frame_embeddings/")

# ──────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".",
                        help="project root (contains audio_embeddings/ and data_splits/)")
    args = parser.parse_args()
    main(args.root)
