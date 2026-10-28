#!/usr/bin/env python3
"""ZEKAD.

The label-free score is

    z = mean(midrank(each of 160 strict-nested-Zentropy scores))
    density(k,beta) = (1-beta) midrank(KNN_k) + beta midrank(LOF_k)
    score(k,beta) = 0.80 density(k,beta) + 0.20 z.

The density channel reproduces the comparison protocol: remove only columns
whose population standard deviation is <= 1e-8 and do not scale.  Native PyOD
KNN/LOF scores are recomputed from raw X; no comparison-result file is read.
Labels are accepted only by the outer experiment wrapper after scores freeze.
Same-dataset best AUC/AP rows are explicitly oracle diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy
import sklearn
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from . import runner as base
except ImportError:
    import runner as base  # type: ignore


METHOD_SHORT_NAME = "ZEKAD"
METHOD_NAME = "ZEKAD"
METHOD_NAME_ZH = "严格嵌套Z熵密度秩融合"
METHOD_VERSION = "3.0.0"
ALPHA = 0.20
DEFAULT_BETA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
DEFAULT_K_GRID = tuple(range(2, 61))
EXPECTED_BASE_SCORE_COUNT = 160
DENSITY_STD_THRESHOLD = 1.0e-8
DENSITY_PROTOCOL = "remove population-std<=1e-8 columns; no scaling; native PyOD KNN/LOF"

DETAIL_FIELDS = [
    "dataset", "config_index", "k", "beta", "alpha", "z_aggregation",
    "parameter_string", "auc", "ap", "n_samples", "n_original_features",
    "n_density_features", "n_removed_features", "base_score_count",
    "model_seconds", "metrics_seconds", "input_sha256", "base_npz_sha256",
]
BEST_FIELDS = [
    "dataset", "status", "error", "selection_metric", "best_metric_value",
    "auc", "ap", "k", "beta", "alpha", "parameter_string",
    "config_index", "n_samples", "n_original_features",
    "n_density_features", "n_removed_features", "n_anomalies",
    "anomaly_rate", "base_score_count", "model_seconds", "metrics_seconds",
    "total_seconds", "input_sha256", "base_npz_sha256", "result_npz",
]
RESULT_FIELDS = ["dataset", "AUC", "AP", "AUC_parameter", "AP_parameter"]
MANIFEST_FIELDS = [
    "dataset", "input_path", "input_sha256", "base_npz_path",
    "base_npz_sha256",
]


def _require_finite(values: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError("%s contains NaN or Inf" % name)


def midrank01(values: np.ndarray) -> np.ndarray:
    """Average tied ranks on a common open-unit scale."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError("cannot rank an empty vector")
    _require_finite(vector, "rank values")
    return (rankdata(vector, method="average") - 0.5) / float(vector.size)


def preprocess_density_features(
    raw_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the exact density-bank preprocessing, deliberately without scaling."""
    data = np.asarray(raw_x)
    if data.ndim != 2 or data.shape[0] < 3 or data.shape[1] < 1:
        raise ValueError("X must contain at least three samples and one feature")
    if np.iscomplexobj(data) or not np.issubdtype(data.dtype, np.number):
        raise ValueError("X must be a real numeric matrix")
    data = np.asarray(data, dtype=np.float64)
    _require_finite(data, "raw features")
    keep_mask = np.std(data, axis=0) > DENSITY_STD_THRESHOLD
    kept = np.flatnonzero(keep_mask).astype(np.int64)
    removed = np.flatnonzero(~keep_mask).astype(np.int64)
    if kept.size == 0:
        raise ValueError("no features remain after the std>1e-8 density filter")
    return np.ascontiguousarray(data[:, kept]), kept, removed


def strict_z_consensus(base_scores: np.ndarray) -> np.ndarray:
    """Mean tied mid-rank across all label-free strict-nested score vectors."""
    scores = np.asarray(base_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] < 3:
        raise ValueError("base scores must be configurations-by-samples")
    _require_finite(scores, "base strict-nested scores")
    consensus = np.mean(np.vstack([midrank01(row) for row in scores]), axis=0)
    _require_finite(consensus, "strict-nested rank consensus")
    return consensus


def _import_pyod_models() -> Tuple[Any, Any, str]:
    try:
        import pyod
        from pyod.models.knn import KNN
        from pyod.models.lof import LOF
    except ImportError as exc:
        raise RuntimeError(
            "ZEKAD requires PyOD 3.2.1; install requirements.txt"
        ) from exc
    return KNN, LOF, str(getattr(pyod, "__version__", "unknown"))


def native_density_bank(
    raw_x: np.ndarray,
    k_grid: Sequence[int] = DEFAULT_K_GRID,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute native PyOD KNN/LOF from X without reading competitor outputs."""
    data, kept, removed = preprocess_density_features(raw_x)
    k_values = sorted(set(int(value) for value in k_grid))
    if not k_values or any(value < 2 for value in k_values):
        raise ValueError("k values must be integers >= 2")
    if max(k_values) >= data.shape[0]:
        raise ValueError("every k must be less than the number of samples")
    KNN, LOF, pyod_version = _import_pyod_models()
    common = dict(
        contamination=0.1, algorithm="auto", leaf_size=30,
        metric="minkowski", p=2, n_jobs=1,
    )
    knn_rows: List[np.ndarray] = []
    lof_rows: List[np.ndarray] = []
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        for k in k_values:
            random.seed(int(seed))
            np.random.seed(int(seed))
            knn = KNN(n_neighbors=k, method="largest", **common).fit(data)
            knn_score = np.asarray(knn.decision_scores_, dtype=np.float64)
            random.seed(int(seed))
            np.random.seed(int(seed))
            lof = LOF(n_neighbors=k, **common).fit(data)
            lof_score = np.asarray(lof.decision_scores_, dtype=np.float64)
            _require_finite(knn_score, "native KNN scores")
            _require_finite(lof_score, "native LOF scores")
            knn_rows.append(knn_score)
            lof_rows.append(lof_score)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    knn_raw = np.vstack(knn_rows)
    lof_raw = np.vstack(lof_rows)
    return {
        "k_values": np.asarray(k_values, dtype=np.int64),
        "knn_raw": knn_raw,
        "lof_raw": lof_raw,
        "knn_midrank": np.vstack([midrank01(row) for row in knn_raw]),
        "lof_midrank": np.vstack([midrank01(row) for row in lof_raw]),
        "kept_feature_indices": kept,
        "removed_feature_indices": removed,
        "pyod_version": pyod_version,
    }


def _parameter_string(k: int, beta: float, base_count: int) -> str:
    return "alpha=0.20,k=%d,beta=%.2f,z=mean%d" % (k, beta, base_count)


def compute_fusion_scores(
    raw_x: np.ndarray,
    base_scores: np.ndarray,
    k_grid: Sequence[int] = DEFAULT_K_GRID,
    beta_grid: Sequence[float] = DEFAULT_BETA_GRID,
    alpha: float = ALPHA,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate the full ZEKAD grid without accepting labels."""
    if not math.isclose(float(alpha), ALPHA, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("ZEKAD v3 fixes alpha at 0.20")
    beta_values = sorted(set(float(value) for value in beta_grid))
    if not beta_values or any(value < 0.0 or value > 1.0 for value in beta_values):
        raise ValueError("beta values must be in [0, 1]")
    z_consensus = strict_z_consensus(base_scores)
    if z_consensus.size != np.asarray(raw_x).shape[0]:
        raise ValueError("base score sample count does not match raw X")
    density = native_density_bank(raw_x, k_grid=k_grid, seed=seed)
    rows: List[np.ndarray] = []
    details: List[Dict[str, Any]] = []
    base_count = int(np.asarray(base_scores).shape[0])
    for k_index, k_value in enumerate(density["k_values"]):
        knn_rank = density["knn_midrank"][k_index]
        lof_rank = density["lof_midrank"][k_index]
        for beta in beta_values:
            density_score = (1.0 - beta) * knn_rank + beta * lof_rank
            score = (1.0 - ALPHA) * density_score + ALPHA * z_consensus
            _require_finite(score, "ZEKAD score")
            details.append({
                "config_index": len(rows), "k": int(k_value),
                "beta": float(beta), "alpha": ALPHA,
                "z_aggregation": "mean_midrank",
                "parameter_string": _parameter_string(
                    int(k_value), float(beta), base_count
                ),
            })
            rows.append(score)
    return {
        "scores": np.vstack(rows), "details": details,
        "z_consensus": z_consensus, **density,
        "base_score_count": base_count, "alpha": ALPHA,
        "beta_grid": np.asarray(beta_values, dtype=np.float64),
        "score_direction": "higher_is_anomaly", "labels_used_by_model": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(str(temporary), str(path))


def _atomic_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(str(temporary), str(path))


def _load_base_scores(
    path: Path, expected_input_hash: str, expected_count: int
) -> Tuple[np.ndarray, str]:
    if not path.is_file():
        raise ValueError("missing strict-nested base NPZ: %s" % path)
    base_npz_hash = _sha256(path)
    with np.load(str(path), allow_pickle=False) as archive:
        if "scores" not in archive.files or "input_sha256" not in archive.files:
            raise ValueError("base NPZ lacks scores or input_sha256")
        scores = np.asarray(archive["scores"], dtype=np.float64)
        saved_hash_array = np.asarray(archive["input_sha256"]).reshape(-1)
        if saved_hash_array.size != 1:
            raise ValueError("base NPZ input hash is malformed")
        saved_hash = str(saved_hash_array[0])
    if saved_hash != expected_input_hash:
        raise ValueError("raw dataset hash disagrees with the base NPZ")
    if scores.ndim != 2 or scores.shape[0] != int(expected_count):
        raise ValueError(
            "expected %d base score vectors but found %s"
            % (int(expected_count), scores.shape)
        )
    _require_finite(scores, "saved strict-nested scores")
    return scores, base_npz_hash


def _best_row(rows: Sequence[Mapping[str, Any]], metric: str) -> Mapping[str, Any]:
    return max(
        rows,
        key=lambda row: (
            float(row[metric]), -int(row["k"]), -float(row["beta"]),
        ),
    )


def evaluate_dataset_file(
    data_path: Path,
    base_npz_path: Path,
    output_dir: Path,
    k_grid: Sequence[int],
    beta_grid: Sequence[float],
    seed: int,
    expected_base_scores: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    started = time.perf_counter()
    input_hash = _sha256(data_path)
    raw_x, labels = base.load_dataset(data_path)
    base_scores, base_npz_hash = _load_base_scores(
        base_npz_path, input_hash, expected_base_scores
    )
    model_started = time.perf_counter()
    grid = compute_fusion_scores(
        raw_x, base_scores, k_grid, beta_grid, ALPHA, seed
    )
    model_seconds = time.perf_counter() - model_started

    # The only label-consuming block: all scores are already frozen.
    metric_started = time.perf_counter()
    auc_values = np.asarray(
        [roc_auc_score(labels, score) for score in grid["scores"]]
    )
    ap_values = np.asarray(
        [average_precision_score(labels, score) for score in grid["scores"]]
    )
    metrics_seconds = time.perf_counter() - metric_started
    detail_rows: List[Dict[str, Any]] = []
    for detail, auc, ap in zip(grid["details"], auc_values, ap_values):
        row = dict(detail)
        row.update({
            "dataset": data_path.stem, "auc": float(auc), "ap": float(ap),
            "n_samples": int(raw_x.shape[0]),
            "n_original_features": int(raw_x.shape[1]),
            "n_density_features": int(grid["kept_feature_indices"].size),
            "n_removed_features": int(grid["removed_feature_indices"].size),
            "base_score_count": int(grid["base_score_count"]),
            "model_seconds": float(model_seconds),
            "metrics_seconds": float(metrics_seconds),
            "input_sha256": input_hash, "base_npz_sha256": base_npz_hash,
        })
        detail_rows.append(row)
    best_auc_detail = _best_row(detail_rows, "auc")
    best_ap_detail = _best_row(detail_rows, "ap")
    result_path = output_dir / "per_dataset" / (data_path.stem + ".npz")
    _atomic_npz(result_path, {
        "labels": labels, "scores": grid["scores"],
        "config_k": np.asarray([row["k"] for row in detail_rows], dtype=np.int64),
        "config_beta": np.asarray(
            [row["beta"] for row in detail_rows], dtype=np.float64
        ),
        "config_alpha": np.full(len(detail_rows), ALPHA),
        "auc": auc_values, "ap": ap_values,
        "z_consensus": grid["z_consensus"],
        "density_k_values": grid["k_values"],
        "knn_raw": grid["knn_raw"], "lof_raw": grid["lof_raw"],
        "knn_midrank": grid["knn_midrank"],
        "lof_midrank": grid["lof_midrank"],
        "density_kept_feature_indices": grid["kept_feature_indices"],
        "density_removed_feature_indices": grid["removed_feature_indices"],
        "input_sha256": np.asarray(input_hash),
        "base_npz_sha256": np.asarray(base_npz_hash),
        "base_score_count": np.asarray(grid["base_score_count"], dtype=np.int64),
        "score_direction": np.asarray("higher_is_anomaly"),
        "labels_used_by_model": np.asarray(False),
    })
    diagnostics = {
        "dataset": data_path.stem, "method_short_name": METHOD_SHORT_NAME,
        "method": METHOD_NAME, "method_name_zh": METHOD_NAME_ZH,
        "method_version": METHOD_VERSION,
        "score_direction": "higher_is_anomaly",
        "score_generation": "label_free", "labels_used_by_model": False,
        "label_informed_score_flip": False,
        "selection_protocol": "separate same-dataset oracle AUC/AP diagnostics",
        "alpha": ALPHA,
        "k_grid": [int(value) for value in grid["k_values"]],
        "beta_grid": [float(value) for value in grid["beta_grid"]],
        "z_aggregation": "mean tied mid-rank of %d strict-nested scores"
        % grid["base_score_count"],
        "density_preprocessing": DENSITY_PROTOCOL,
        "pyod_version": grid["pyod_version"],
        "input_sha256": input_hash, "base_npz_sha256": base_npz_hash,
        "result_npz": "per_dataset/%s.npz" % data_path.stem,
    }
    _atomic_json(
        output_dir / "per_dataset" / (data_path.stem + "_diagnostics.json"),
        diagnostics,
    )
    total_seconds = time.perf_counter() - started

    def make_best(detail: Mapping[str, Any], metric: str) -> Dict[str, Any]:
        return {
            "dataset": data_path.stem, "status": "ok", "error": "",
            "selection_metric": metric,
            "best_metric_value": float(detail[metric]),
            "auc": float(detail["auc"]), "ap": float(detail["ap"]),
            "k": int(detail["k"]), "beta": float(detail["beta"]),
            "alpha": ALPHA, "parameter_string": detail["parameter_string"],
            "config_index": int(detail["config_index"]),
            "n_samples": int(raw_x.shape[0]),
            "n_original_features": int(raw_x.shape[1]),
            "n_density_features": int(grid["kept_feature_indices"].size),
            "n_removed_features": int(grid["removed_feature_indices"].size),
            "n_anomalies": int(np.sum(labels)),
            "anomaly_rate": float(np.mean(labels)),
            "base_score_count": int(grid["base_score_count"]),
            "model_seconds": float(model_seconds),
            "metrics_seconds": float(metrics_seconds),
            "total_seconds": float(total_seconds),
            "input_sha256": input_hash, "base_npz_sha256": base_npz_hash,
            "result_npz": "per_dataset/%s.npz" % data_path.stem,
        }
    return (
        make_best(best_auc_detail, "auc"),
        make_best(best_ap_detail, "ap"), detail_rows, diagnostics,
    )


def _error_best(dataset: str, metric: str, error: str) -> Dict[str, Any]:
    row = {field: "" for field in BEST_FIELDS}
    row.update({
        "dataset": dataset, "status": "error", "error": error,
        "selection_metric": metric,
    })
    return row


def _write_outputs(
    output_dir: Path,
    metadata: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    best_auc_rows: Sequence[Mapping[str, Any]],
    best_ap_rows: Sequence[Mapping[str, Any]],
    detail_rows: Sequence[Mapping[str, Any]],
    completed: Sequence[str],
    status: str,
) -> None:
    _atomic_csv(output_dir / "dataset_manifest.csv", manifest_rows, MANIFEST_FIELDS)
    _atomic_csv(output_dir / "grid_detail_auc_ap.csv", detail_rows, DETAIL_FIELDS)
    _atomic_csv(output_dir / "best_auc_summary.csv", best_auc_rows, BEST_FIELDS)
    _atomic_csv(output_dir / "best_ap_summary.csv", best_ap_rows, BEST_FIELDS)
    _atomic_csv(output_dir / "best_by_auc.csv", best_auc_rows, BEST_FIELDS)
    _atomic_csv(output_dir / "best_by_ap.csv", best_ap_rows, BEST_FIELDS)
    ap_by_dataset = {str(row["dataset"]): row for row in best_ap_rows}
    result_rows: List[Dict[str, Any]] = []
    for auc_row in best_auc_rows:
        dataset = str(auc_row["dataset"])
        ap_row = ap_by_dataset.get(dataset, {})
        if auc_row.get("status") == "ok" and ap_row.get("status") == "ok":
            result_rows.append({
                "dataset": dataset, "AUC": auc_row["auc"], "AP": ap_row["ap"],
                "AUC_parameter": auc_row["parameter_string"],
                "AP_parameter": ap_row["parameter_string"],
            })
    _atomic_csv(output_dir / "SNZE_DRF_AUC_AP.csv", result_rows, RESULT_FIELDS)
    payload = {
        "status": status, "metadata": dict(metadata),
        "completed_datasets": list(completed),
        "best_auc_summary": list(best_auc_rows),
        "best_ap_summary": list(best_ap_rows), "grid_detail": list(detail_rows),
    }
    _atomic_json(output_dir / "checkpoint.json", payload)
    _atomic_json(output_dir / "run_results.json", payload)
    successful_auc = [row for row in best_auc_rows if row.get("status") == "ok"]
    successful_ap = [row for row in best_ap_rows if row.get("status") == "ok"]
    overview = {
        "method_short_name": METHOD_SHORT_NAME, "method": METHOD_NAME,
        "method_name_zh": METHOD_NAME_ZH,
        "method_version": METHOD_VERSION, "status": status,
        "successful_datasets": len(result_rows),
        "failed_datasets": len(best_auc_rows) - len(successful_auc),
        "mean_oracle_best_auc": (
            float(np.mean([float(row["auc"]) for row in successful_auc]))
            if successful_auc else None
        ),
        "mean_oracle_best_ap": (
            float(np.mean([float(row["ap"]) for row in successful_ap]))
            if successful_ap else None
        ),
        "score_direction": "higher_is_anomaly",
        "score_generation": "label_free",
        "selection_protocol": "separate same-dataset oracle AUC/AP diagnostics",
        "alpha": ALPHA,
    }
    _atomic_json(output_dir / "overview.json", overview)


def _snapshot_sources(output_dir: Path) -> None:
    destination = output_dir / "source_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parent
    for name in (
        "runner_improved.py",
        "runner.py",
        "requirements.txt",
        "SNZE_DRF_Density_Equivalence.md",
    ):
        source = source_root / name
        if source.is_file():
            shutil.copy2(str(source), str(destination / name))


def run_batch(args: argparse.Namespace) -> int:
    files = sorted(Path(args.data_dir).glob("*.mat"))
    if not files:
        raise ValueError("no .mat files found under %s" % args.data_dir)
    if args.limit is None and len(files) != int(args.expected_datasets):
        raise ValueError(
            "expected %d datasets but found %d"
            % (int(args.expected_datasets), len(files))
        )
    if args.limit is not None:
        files = files[: int(args.limit)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists() and not args.resume and not args.overwrite:
        raise ValueError("checkpoint exists; use --resume, --overwrite, or a new directory")
    base_run_dir = Path(args.base_run_dir)
    base_config_path = base_run_dir / "run_config.json"
    base_config: Dict[str, Any] = {}
    if base_config_path.is_file():
        with base_config_path.open("r", encoding="utf-8") as handle:
            base_config = json.load(handle)
        if base_config.get("score_direction") not in (None, "higher_is_anomaly"):
            raise ValueError("base score direction is not higher_is_anomaly")
        if base_config.get("label_informed_score_flip") not in (None, False):
            raise ValueError("base run reports label-informed score flipping")

    manifest_rows: List[Dict[str, Any]] = []
    for path in files:
        base_npz = base_run_dir / "per_dataset" / (path.stem + ".npz")
        if not base_npz.is_file():
            raise ValueError("missing base result for %s" % path.stem)
        manifest_rows.append({
            "dataset": path.stem, "input_path": str(path.resolve()),
            "input_sha256": _sha256(path),
            "base_npz_path": str(base_npz.resolve()),
            "base_npz_sha256": _sha256(base_npz),
        })
    _, _, pyod_version = _import_pyod_models()
    metadata: Dict[str, Any] = {
        "method_short_name": METHOD_SHORT_NAME, "method": METHOD_NAME,
        "method_name_zh": METHOD_NAME_ZH,
        "method_version": METHOD_VERSION,
        "data_dir": str(Path(args.data_dir).resolve()),
        "base_run_dir": str(base_run_dir.resolve()),
        "base_method": base_config.get("method"),
        "k_grid": [int(value) for value in args.k_grid],
        "beta_grid": [float(value) for value in args.beta_grid],
        "alpha": ALPHA,
        "z_aggregation": "mean tied mid-rank of %d strict-nested score vectors"
        % int(args.expected_base_scores),
        "density_preprocessing": DENSITY_PROTOCOL, "seed": int(args.seed),
        "score_direction": "higher_is_anomaly", "score_generation": "label_free",
        "labels_used_by_model": False, "label_informed_score_flip": False,
        "selection_protocol": "separate same-dataset oracle AUC/AP diagnostics",
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "pyod": pyod_version,
        "platform": platform.platform(), "command": sys.argv,
        "started_unix": time.time(),
    }
    best_auc_rows: List[Dict[str, Any]] = []
    best_ap_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    completed: List[str] = []
    if args.resume and checkpoint_path.is_file():
        with checkpoint_path.open("r", encoding="utf-8") as handle:
            prior = json.load(handle)
        prior_metadata = prior.get("metadata", {})
        for key in ("k_grid", "beta_grid", "alpha", "seed"):
            if prior_metadata.get(key) != metadata.get(key):
                raise ValueError("resume setting changed: %s" % key)
        best_auc_rows = list(prior.get("best_auc_summary", []))
        best_ap_rows = list(prior.get("best_ap_summary", []))
        detail_rows = list(prior.get("grid_detail", []))
        completed = list(prior.get("completed_datasets", []))
    _atomic_json(output_dir / "run_config.json", metadata)
    _atomic_json(output_dir / "environment.json", {
        "python": platform.python_version(), "python_executable": sys.executable,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "pyod": pyod_version,
        "platform": platform.platform(),
    })
    _snapshot_sources(output_dir)
    log_path = output_dir / "run.log"
    if not args.resume or not log_path.exists():
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("%s (%s) run\n" % (METHOD_SHORT_NAME, METHOD_NAME))

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    completed_set = set(completed)
    run_started = time.perf_counter()
    manifest_by_name = {str(row["dataset"]): row for row in manifest_rows}
    for position, path in enumerate(files, start=1):
        if path.stem in completed_set:
            log("[%d/%d] %s (already completed)" % (position, len(files), path.name))
            continue
        best_auc_rows = [row for row in best_auc_rows if row.get("dataset") != path.stem]
        best_ap_rows = [row for row in best_ap_rows if row.get("dataset") != path.stem]
        detail_rows = [row for row in detail_rows if row.get("dataset") != path.stem]
        log("[%d/%d] %s" % (position, len(files), path.name))
        succeeded = False
        try:
            manifest = manifest_by_name[path.stem]
            best_auc, best_ap, rows, _ = evaluate_dataset_file(
                path, Path(str(manifest["base_npz_path"])), output_dir,
                args.k_grid, args.beta_grid, int(args.seed),
                int(args.expected_base_scores),
            )
            best_auc_rows.append(best_auc)
            best_ap_rows.append(best_ap)
            detail_rows.extend(rows)
            completed.append(path.stem)
            completed_set.add(path.stem)
            succeeded = True
            log("  AUC=%.6f (%s) | AP=%.6f (%s)" % (
                best_auc["auc"], best_auc["parameter_string"],
                best_ap["ap"], best_ap["parameter_string"],
            ))
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
            best_auc_rows.append(_error_best(path.stem, "auc", error))
            best_ap_rows.append(_error_best(path.stem, "ap", error))
            log("  ERROR: %s" % error)
        metadata["elapsed_seconds"] = float(time.perf_counter() - run_started)
        metadata["successful_datasets"] = len(completed)
        _atomic_json(output_dir / "run_config.json", metadata)
        _write_outputs(
            output_dir, metadata, manifest_rows, best_auc_rows, best_ap_rows,
            detail_rows, completed,
            "running" if succeeded else "running_with_errors",
        )
    metadata["elapsed_seconds"] = float(time.perf_counter() - run_started)
    metadata["finished_unix"] = time.time()
    failed = sum(row.get("status") != "ok" for row in best_auc_rows)
    final_status = "complete" if failed == 0 else "complete_with_errors"
    _atomic_json(output_dir / "run_config.json", metadata)
    _write_outputs(
        output_dir, metadata, manifest_rows, best_auc_rows, best_ap_rows,
        detail_rows, completed, final_status,
    )
    log("Completed %d datasets in %.2fs" % (len(files), metadata["elapsed_seconds"]))
    return 0 if failed == 0 else 1


def _parse_k_grid(text: str) -> List[int]:
    try:
        if ":" in text and "," not in text:
            start, stop = [int(part.strip()) for part in text.split(":", 1)]
            values = list(range(start, stop + 1))
        else:
            values = sorted(set(int(p.strip()) for p in text.split(",") if p.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("k grid must contain integers") from exc
    if not values or any(item < 2 for item in values):
        raise argparse.ArgumentTypeError("k values must be >= 2")
    return values


def _parse_beta_grid(text: str) -> List[float]:
    try:
        values = sorted(set(float(p.strip()) for p in text.split(",") if p.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("beta grid must contain numbers") from exc
    if not values or any(item < 0.0 or item > 1.0 for item in values):
        raise argparse.ArgumentTypeError("beta values must be in [0, 1]")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-grid", type=_parse_k_grid, default=list(DEFAULT_K_GRID))
    parser.add_argument(
        "--beta-grid", type=_parse_beta_grid, default=list(DEFAULT_BETA_GRID)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-base-scores", type=int, default=160)
    parser.add_argument("--expected-datasets", type=int, default=70)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_base_scores < 1:
        parser.error("--expected-base-scores must be positive")
    if args.expected_datasets < 1:
        parser.error("--expected-datasets must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        return run_batch(args)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
