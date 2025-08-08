import argparse
import pandas as pd
import json
import os


def load_split_ids(split_file, split_name):
    with open(split_file) as f:
        data = json.load(f)
    return set(data[split_name])


def main():
    parser = argparse.ArgumentParser(description="Evaluate UTD split accuracy from predictions")
    parser.add_argument("--pred_csv", type=str, required=True, help="Path to csv with predictions")
    parser.add_argument("--splits_path", type=str, required=True, help="Path to  JSON file with splits")
    parser.add_argument("--split_name", type=str, required=True, help="Name of the split in JSON (e.g., debiased-balanced)")

    args = parser.parse_args()

    if not os.path.exists(args.pred_csv):
        raise FileNotFoundError(f"Prediction CSV not found: {args.pred_csv}")
    if not os.path.exists(args.splits_path):
        raise FileNotFoundError(f"Split file not found: {args.splits_path}")

    pred = pd.read_csv(args.pred_csv)
    with open(args.splits_path) as f:
        data = json.load(f)
    video_ids = set(data[args.split_name])

    pred_filtered = pred[pred['video_id'].isin(video_ids)]
    acc_top1 = pred_filtered['acc_top1'].mean() * 100 if 'acc_top1' in pred.columns else None
    acc_top5 = pred_filtered['acc_top5'].mean() * 100 if 'acc_top5' in pred.columns else None

    print(f"Results for split '{args.split_name}':")
    if acc_top1 is not None:
        print(f"Top-1 Accuracy: {acc_top1:.2f}%")
    else:
        print("Top-1 Accuracy: not available")

    if acc_top5 is not None:
        print(f"Top-5 Accuracy: {acc_top5:.2f}%")
    else:
        print("Top-5 Accuracy: not available")


if __name__ == "__main__":
    main()