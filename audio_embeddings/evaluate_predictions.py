#!/usr/bin/env python3
"""
evaluate_predictions.py

usage:
    python evaluate_predictions.py \
        --pred predictions.csv \
        --json sarcasm_data.json
"""
import argparse, json, csv, pathlib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main(args):
    # ---------- ground-truth labels ---------------------------------
    with open(args.json) as f:
        gold_map = {
            k: 1 if v["sarcasm"] else 0
            for k, v in json.load(f).items()
        }

    # ---------- predictions & confidences ---------------------------
    gold, pred = [], []
    with open(args.pred) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            sample_id = row["id"].replace("_embedding", "")  # strip suffix
            if sample_id not in gold_map:
                print(f"[warn] {sample_id} not in JSON – skipped")
                continue
            pred.append(int(row["prediction"]))
            gold.append(gold_map[sample_id])

    if not gold:
        raise RuntimeError("No overlapping IDs between JSON and CSV!")

    # ---------- metrics ---------------------------------------------
    acc  = accuracy_score(gold, pred)
    prec = precision_score(gold, pred, zero_division=0)
    rec  = recall_score(gold, pred, zero_division=0)
    f1   = f1_score(gold, pred, zero_division=0)

    print(f"Samples evaluated : {len(gold)}")
    print(f"Accuracy          : {acc:.3f}")
    print(f"Precision         : {prec:.3f}")
    print(f"Recall            : {rec:.3f}")
    print(f"F1 score          : {f1:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True,
                    help="CSV produced by predict_from_embeddings.py")
    ap.add_argument("--json", required=True,
                    help="sarcasm_data.json containing ground-truth labels")
    main(ap.parse_args())
