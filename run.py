#!/usr/bin/env python3
"""Run ZEKAD on a MATLAB file or a directory of MATLAB files."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import MatReadError
from sklearn.metrics import average_precision_score, roc_auc_score

from zekad import score_grid, score_samples


def load_data(path):
    """Read labeled trandata or an unlabeled feature matrix named X."""
    try:
        content = loadmat(str(path))
    except NotImplementedError as exc:
        try:
            import h5py
        except ImportError:
            raise ValueError("MATLAB v7.3 files require optional h5py") from exc
        with h5py.File(str(path), "r") as handle:
            content = {key: np.asarray(handle[key]).T
                       for key in ("trandata", "X") if key in handle}
    key = "trandata" if "trandata" in content else "X"
    if key not in content:
        raise ValueError("expected a matrix named trandata or X")
    matrix = np.asarray(content[key])
    if (matrix.ndim != 2 or matrix.shape[0] < 3
            or matrix.shape[1] < (2 if key == "trandata" else 1)):
        raise ValueError("input matrix has an invalid shape")
    if np.iscomplexobj(matrix) or not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("input matrix must contain real numbers")
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("input matrix contains NaN or infinity")
    if key == "X":
        return matrix, None
    labels = matrix[:, -1]
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("trandata labels must contain both 0 and 1")
    return matrix[:, :-1], labels.astype(np.int8)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help=".mat file or directory")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--k", type=int, help="fixed density neighbors (default: 5)")
    parser.add_argument("--beta", type=float, help="fixed LOF weight (default: 0)")
    parser.add_argument("--grid", action="store_true", help=(
        "generate all 295 candidates; label-based maxima are oracle results"))
    args = parser.parse_args(argv)
    if args.grid and (args.k is not None or args.beta is not None):
        parser.error("--k and --beta apply only to fixed-parameter runs")
    k = 5 if args.k is None else args.k
    beta = 0.0 if args.beta is None else args.beta
    if k < 2 or not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
        parser.error("k must be >= 2 and beta must be in [0, 1]")
    if args.input.is_dir():
        files = sorted(path for path in args.input.iterdir()
                       if path.is_file() and path.suffix.lower() == ".mat")
    elif args.input.is_file() and args.input.suffix.lower() == ".mat":
        files = [args.input]
    else:
        parser.error("input must be an existing .mat file or directory")
    if not files:
        parser.error("no .mat files found")
    if len({path.stem for path in files}) != len(files):
        parser.error("dataset file stems must be unique")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["dataset", "k", "beta", "AUC", "AP"])
            for path in files:
                features, labels = load_data(path)
                if args.grid:
                    result = score_grid(features, dataset_name=path.stem)
                else:
                    scores = score_samples(features, dataset_name=path.stem, k=k, beta=beta)
                    result = {"scores": scores[None, :],
                              "config_k": np.asarray([k]),
                              "config_beta": np.asarray([beta])}
                # Scoring is complete before labels enter any evaluation.
                saved = {key: result[key] for key in ("scores", "config_k", "config_beta")}
                if "z_consensus" in result:
                    saved["z_consensus"] = result["z_consensus"]
                if labels is not None:
                    saved["labels"] = labels
                np.savez_compressed(str(args.output_dir / (path.stem + ".npz")), **saved)
                rows = []
                for score, neighbors, weight in zip(result["scores"], result["config_k"],
                                                     result["config_beta"]):
                    auc = roc_auc_score(labels, score) if labels is not None else ""
                    ap = average_precision_score(labels, score) if labels is not None else ""
                    row = [path.stem, int(neighbors), float(weight), auc, ap]
                    writer.writerow(row)
                    rows.append(row)
                handle.flush()
                print("%s: saved %d score vector(s)" % (path.stem, len(rows)), flush=True)
                if labels is not None:
                    for index, metric in ((3, "AUC"), (4, "AP")):
                        best = max(rows, key=lambda row: (row[index], -row[1], -row[2]))
                        print("  %s%s=%.6f (k=%d, beta=%g)" % (
                            "oracle " if args.grid else "", metric,
                            best[index], best[1], best[2]), flush=True)
    except (ValueError, RuntimeError, OSError, MatReadError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
