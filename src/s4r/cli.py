"""End-to-end Route C entry point.

Usage:
    uv run python -m s4r.cli route-c \
        --features data/processed/village_features.csv \
        --sample data/raw/Sample_submission_file.csv \
        --out outputs/submission_routeC.csv \
        [--baseline-only] [--weak-labels data/weak_labels/annotations.csv] \
        [--alpha 0.38] [--seed 0] [--restarts 8] [--allow-synthetic]
"""

import argparse
import sys

from s4r import config
from s4r.data.ingest import DataValidationError, load_features, load_sample_submission, standardize_features
from s4r.fallback.baseline import baseline_allocation
from s4r.fallback.train import TrainConfig, train
from s4r.features.coverage import coverage_confidence
from s4r.submission.writer import SubmissionError, comparison_report, report_summary, write_submission
from s4r.weak_labels.ingest import WeakLabelError, load_weak_labels


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="s4r")
    sub = p.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("extract", help="regenerate village_features.csv from raw Capella data")
    ex.add_argument("--data-dir", required=True, help="competition data directory")
    ex.add_argument("--out", required=True, help="output CSV path")

    rc = sub.add_parser("route-c", help="classical LLP-constrained fallback pipeline")
    rc.add_argument("--features", required=True)
    rc.add_argument("--sample", required=True)
    rc.add_argument("--out", required=True)
    rc.add_argument("--baseline-only", action="store_true", help="hedged regional-mean allocation, no training")
    rc.add_argument("--weak-labels", default=None)
    rc.add_argument("--alpha", type=float, default=config.ALPHA_CAP)
    rc.add_argument("--seed", type=int, default=0)
    rc.add_argument("--restarts", type=int, default=8)
    rc.add_argument("--allow-synthetic", action="store_true")
    rc.add_argument("--run-dir", default="experiments/runs")
    return p


def run_route_c(args) -> int:
    features_df = load_features(args.features)
    sample_df = load_sample_submission(args.sample)
    area = features_df["area_ha"].to_numpy()

    preds = {"baseline": baseline_allocation(area)}

    if args.baseline_only:
        final = preds["baseline"]
    else:
        X = standardize_features(features_df)
        conf = coverage_confidence(features_df)
        anchors = None
        if args.weak_labels:
            anchors = load_weak_labels(args.weak_labels, features_df)
        cfg = TrainConfig(alpha=args.alpha, seed=args.seed, n_restarts=args.restarts)
        result = train(X, area, conf, cfg, anchors=anchors, run_dir=args.run_dir)
        preds["route_c"] = result["pred"]
        final = result["pred"]
        print(f"training loss: {result['loss']:.6f}")
        print(f"loss components: {result['loss_components']}")
        if result["run_log_path"]:
            print(f"run log: {result['run_log_path']}")

    print("\n=== aggregate summary (grand_total / per-crop shares vs targets) ===")
    print(report_summary(preds).to_string(index=False))
    print("\n=== per-village allocation ===")
    print(comparison_report(preds, features_df, args.alpha).to_string(index=False))

    out = write_submission(final, features_df, sample_df, args.out, allow_synthetic=args.allow_synthetic)
    print(f"\nsubmission written: {out}")
    if features_df["is_synthetic"].any():
        print("WARNING: predictions derive from SYNTHETIC data — do NOT submit this file.")
    return 0


def run_extract(args) -> int:
    from s4r.features.extract import extract_features

    df = extract_features(args.data_dir, out_csv=args.out)
    n_zero = int((df[[f"coverage_{d}" for d in config.DATES]].sum(axis=1) == 0).sum())
    print(f"extracted {len(df)} villages -> {args.out}")
    print(f"area sum: {df['area_ha'].sum():.2f} ha; zero-coverage villages: {n_zero}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "extract":
            return run_extract(args)
        return run_route_c(args)
    except (DataValidationError, SubmissionError, WeakLabelError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
