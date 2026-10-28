#!/usr/bin/env python3
"""Run ZEKAD structural branch on MATLAB anomaly datasets.

The model is label free.  For every B view (a feature or a random feature
subspace), T is the complement inside the current feature universe.  Gaussian
relations on B and T induce the three support states common/B-only/T-only.
The implementation verifies, to 1e-12 tolerance, the chain-rule identity

    Z_flat = H(fine ball) + E[H(support state | fine ball)]
           = H(coarse ball) + E[H(fine ball | coarse ball)
                                  + E[H(support state | fine ball)]]
           = Z_nested.

Raw Z is saved only as a mathematical/audit quantity.  Feature selection and
subspace fusion use normalized mutual information between fine-ball identity
and the three-state distribution.  Labels are accepted only by the experiment
wrapper for ROC-AUC and average precision; score direction is never flipped.

This file intentionally supports Python 3.8 and has no dependency on the older
GBZentropy or GBFRZAD project directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import itertools
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import scipy
import sklearn
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, roc_auc_score


EPS = 1.0e-12
RELATION_FLOOR = 1.0e-15
# The saved-result validator uses this as a hard absolute contract.  Keeping
# one tolerance here avoids a model run succeeding under a looser relative
# comparison and then failing its independent audit.
IDENTITY_TOL = 1.0e-12


@dataclass(frozen=True)
class ModelConfig:
    """Fixed, label-independent settings for the detector."""

    seed: int = 20260917
    max_balls: Optional[int] = None
    relation_floor: float = RELATION_FLOOR


@dataclass
class PreprocessState:
    original_feature_indices: np.ndarray
    dropped_feature_indices: np.ndarray
    medians: np.ndarray
    scales: np.ndarray
    n_original_features: int


@dataclass
class TreeNode:
    parent: int
    depth: int
    indices: np.ndarray
    center: np.ndarray
    sse: float
    left: int = -1
    right: int = -1


@dataclass
class BallHierarchy:
    nodes: List[TreeNode]
    accepted_splits: np.ndarray
    fine_node_ids: np.ndarray
    coarse_node_ids: np.ndarray
    fine_membership: np.ndarray
    coarse_membership: np.ndarray
    fine_to_coarse: np.ndarray
    fine_centers: np.ndarray
    fine_sizes: np.ndarray
    coarse_sizes: np.ndarray


@dataclass
class NestedZResult:
    probabilities: np.ndarray
    local_entropy: np.ndarray
    information: np.ndarray
    z_flat: float
    z_nested: float
    z_joint: float
    nested_error: float
    marginal_probabilities: np.ndarray
    marginal_entropy: float
    conditional_entropy: float
    relevance: float
    support_recursive_entropy: np.ndarray
    support_recursive_error: float
    probability_sum_max_error: float
    probability_min: float
    probability_max: float
    b_features: np.ndarray
    t_features: np.ndarray
    complement_degenerate: bool


@dataclass
class SubspaceCache:
    local_features: np.ndarray
    complement_features: np.ndarray
    nested: NestedZResult
    centers: np.ndarray
    radii: np.ndarray
    distances: np.ndarray
    neighbor_order: np.ndarray
    within_deviation: np.ndarray


@dataclass
class RhoArtifact:
    selected_count: int
    selected_local: np.ndarray
    selected_original: np.ndarray
    hierarchy: BallHierarchy
    subspaces: List[np.ndarray]
    caches: List[SubspaceCache]
    sigma_squared: float
    path_utility: float


def _require_finite(values: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError("%s contains NaN or Inf" % name)


def _entropy(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    logs = np.zeros_like(p)
    np.log2(p, out=logs, where=p > 0.0)
    return -np.sum(p * logs, axis=axis)


def preprocess_features(x: np.ndarray) -> Tuple[np.ndarray, PreprocessState]:
    """Drop constant columns and median/IQR scale finite numeric features."""
    data = np.asarray(x)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 1:
        raise ValueError("X must contain at least two samples and one feature")
    if np.iscomplexobj(data) or not np.issubdtype(data.dtype, np.number):
        raise ValueError("X must be a real numeric matrix")
    data = np.asarray(data, dtype=np.float64)
    _require_finite(data, "features")

    medians = np.median(data, axis=0)
    q25, q75 = np.percentile(data, [25.0, 75.0], axis=0)
    scales = q75 - q25
    mad = 1.4826 * np.median(np.abs(data - medians), axis=0)
    scales = np.where(scales > EPS, scales, mad)
    standard_deviation = np.std(data, axis=0)
    scales = np.where(scales > EPS, scales, standard_deviation)
    data_range = np.max(data, axis=0) - np.min(data, axis=0)
    # "Constant" is an exact data property; do not silently discard a valid
    # low-amplitude feature merely because its raw range is below EPS.
    active = np.flatnonzero(data_range > 0.0).astype(np.int64)
    dropped = np.flatnonzero(data_range == 0.0).astype(np.int64)
    if active.size == 0:
        raise ValueError("all feature columns are constant")
    if np.any(scales[active] <= 0.0) or not np.all(np.isfinite(scales[active])):
        raise ValueError("an active feature has an invalid scale")
    scaled = (data[:, active] - medians[active]) / scales[active]
    scaled = np.ascontiguousarray(scaled, dtype=np.float64)
    _require_finite(scaled, "scaled features")
    return scaled, PreprocessState(
        original_feature_indices=active,
        dropped_feature_indices=dropped,
        medians=np.asarray(medians, dtype=np.float64),
        scales=np.asarray(scales, dtype=np.float64),
        n_original_features=int(data.shape[1]),
    )


def _center_sse(x: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, float]:
    points = x[indices]
    center = np.mean(points, axis=0)
    difference = points - center
    sse = float(np.einsum("ij,ij->", difference, difference))
    if not np.isfinite(sse):
        raise ValueError("non-finite granular-ball SSE")
    return center, sse


def _candidate_split(
    x: np.ndarray, indices: np.ndarray
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """Return a positive-BIC deterministic farthest-pair split."""
    count = int(indices.size)
    dimensions = int(x.shape[1])
    if count < 2:
        return None
    points = x[indices]
    center = np.mean(points, axis=0)
    distance_to_center = np.einsum("ij,ij->i", points - center, points - center)
    position_a = int(np.argmax(distance_to_center))
    seed_a = points[position_a]
    distance_a = np.einsum("ij,ij->i", points - seed_a, points - seed_a)
    position_b = int(np.argmax(distance_a))
    if position_b == position_a or distance_a[position_b] <= EPS:
        return None
    seed_b = points[position_b]
    distance_b = np.einsum("ij,ij->i", points - seed_b, points - seed_b)
    left_mask = distance_a <= distance_b
    if bool(np.all(left_mask)) or bool(np.all(~left_mask)):
        return None
    left = indices[left_mask]
    right = indices[~left_mask]
    _, sse_one = _center_sse(x, indices)
    _, sse_left = _center_sse(x, left)
    _, sse_right = _center_sse(x, right)
    sse_two = sse_left + sse_right
    p_one = dimensions + 1
    p_two = 2 * dimensions + 2
    bic_one = (
        count * dimensions * math.log(sse_one / (count * dimensions) + EPS)
        + p_one * math.log(count)
    )
    bic_two = (
        count * dimensions * math.log(sse_two / (count * dimensions) + EPS)
        + p_two * math.log(count)
    )
    improvement = bic_one - bic_two
    if not np.isfinite(improvement) or improvement <= 0.0:
        return None
    return float(improvement), left, right


def _is_ancestor(nodes: Sequence[TreeNode], ancestor: int, node_id: int) -> bool:
    current = int(node_id)
    while current >= 0:
        if current == int(ancestor):
            return True
        current = int(nodes[current].parent)
    return False


def build_hierarchical_balls(
    x: np.ndarray, max_balls: Optional[int] = None
) -> BallHierarchy:
    """Build a BIC tree and derive a valid coarse frontier from its split path."""
    data = np.asarray(x, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] < 1:
        raise ValueError("ball input must be a non-empty two-dimensional matrix")
    _require_finite(data, "ball input")
    sample_count = int(data.shape[0])
    automatic = max(1, int(math.ceil(math.sqrt(sample_count))))
    budget = automatic if max_balls is None else int(max_balls)
    budget = max(1, min(budget, sample_count))

    root_indices = np.arange(sample_count, dtype=np.int64)
    root_center, root_sse = _center_sse(data, root_indices)
    nodes = [TreeNode(-1, 0, root_indices, root_center, root_sse)]
    leaves: Set[int] = {0}
    accepted_splits: List[int] = []
    heap: List[Tuple[float, int, np.ndarray, np.ndarray]] = []
    candidate = _candidate_split(data, root_indices)
    if candidate is not None:
        improvement, left, right = candidate
        heapq.heappush(heap, (-improvement, 0, left, right))

    while heap and len(leaves) < budget:
        _, node_id, left, right = heapq.heappop(heap)
        if node_id not in leaves:
            continue
        parent = nodes[node_id]
        left_center, left_sse = _center_sse(data, left)
        right_center, right_sse = _center_sse(data, right)
        left_id = len(nodes)
        nodes.append(
            TreeNode(node_id, parent.depth + 1, left, left_center, left_sse)
        )
        right_id = len(nodes)
        nodes.append(
            TreeNode(node_id, parent.depth + 1, right, right_center, right_sse)
        )
        parent.left = left_id
        parent.right = right_id
        leaves.remove(node_id)
        leaves.add(left_id)
        leaves.add(right_id)
        accepted_splits.append(node_id)
        for child_id in (left_id, right_id):
            child_candidate = _candidate_split(data, nodes[child_id].indices)
            if child_candidate is not None:
                improvement, child_left, child_right = child_candidate
                heapq.heappush(
                    heap,
                    (-improvement, child_id, child_left, child_right),
                )

    fine_node_ids = np.asarray(
        sorted(leaves, key=lambda item: int(np.min(nodes[item].indices))),
        dtype=np.int64,
    )
    fine_count = int(fine_node_ids.size)
    coarse_target = (
        1
        if fine_count == 1
        else min(fine_count, max(2, int(math.ceil(math.sqrt(fine_count)))))
    )
    frontier: Set[int] = {0}
    for split_node_id in accepted_splits[: max(0, coarse_target - 1)]:
        if split_node_id not in frontier:
            raise RuntimeError("accepted split sequence is not a valid tree frontier")
        frontier.remove(split_node_id)
        frontier.add(nodes[split_node_id].left)
        frontier.add(nodes[split_node_id].right)
    if len(frontier) != coarse_target:
        raise RuntimeError("failed to derive the requested coarse tree cut")
    coarse_node_ids = np.asarray(
        sorted(frontier, key=lambda item: int(np.min(nodes[item].indices))),
        dtype=np.int64,
    )
    coarse_lookup = {int(node_id): position for position, node_id in enumerate(coarse_node_ids)}

    fine_membership = np.empty(sample_count, dtype=np.int64)
    fine_to_coarse = np.empty(fine_count, dtype=np.int64)
    fine_centers = np.empty((fine_count, data.shape[1]), dtype=np.float64)
    fine_sizes = np.empty(fine_count, dtype=np.int64)
    for fine_position, fine_node_value in enumerate(fine_node_ids):
        fine_node = int(fine_node_value)
        fine_membership[nodes[fine_node].indices] = fine_position
        fine_centers[fine_position] = nodes[fine_node].center
        fine_sizes[fine_position] = int(nodes[fine_node].indices.size)
        ancestor = fine_node
        while ancestor not in coarse_lookup:
            ancestor = int(nodes[ancestor].parent)
            if ancestor < 0:
                raise RuntimeError("fine leaf has no ancestor in the coarse cut")
        fine_to_coarse[fine_position] = coarse_lookup[ancestor]
        if not _is_ancestor(nodes, ancestor, fine_node):
            raise RuntimeError("invalid fine-to-coarse descendant mapping")
    coarse_membership = fine_to_coarse[fine_membership]
    coarse_sizes = np.bincount(
        fine_to_coarse, weights=fine_sizes, minlength=coarse_node_ids.size
    ).astype(np.int64)
    if not np.array_equal(
        np.bincount(fine_membership, minlength=fine_count), fine_sizes
    ):
        raise RuntimeError("fine membership counts disagree with fine sizes")
    if not np.array_equal(
        np.bincount(coarse_membership, minlength=coarse_node_ids.size), coarse_sizes
    ):
        raise RuntimeError("coarse membership counts disagree with coarse sizes")
    return BallHierarchy(
        nodes=nodes,
        accepted_splits=np.asarray(accepted_splits, dtype=np.int64),
        fine_node_ids=fine_node_ids,
        coarse_node_ids=coarse_node_ids,
        fine_membership=fine_membership,
        coarse_membership=coarse_membership,
        fine_to_coarse=fine_to_coarse,
        fine_centers=fine_centers,
        fine_sizes=fine_sizes,
        coarse_sizes=coarse_sizes,
    )


def _pairwise_total_squared(centers: np.ndarray) -> np.ndarray:
    difference = centers[:, None, :] - centers[None, :, :]
    total = np.sum(difference * difference, axis=2)
    np.maximum(total, 0.0, out=total)
    return total


def _bandwidth_squared(total_squared: np.ndarray, feature_count: int) -> float:
    mean_squared = total_squared / max(int(feature_count), 1)
    upper = mean_squared[np.triu_indices(mean_squared.shape[0], 1)]
    positive = upper[upper > 0.0]
    sigma_squared = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(sigma_squared) or sigma_squared <= 0.0:
        raise ValueError("computed Gaussian bandwidth is not finite and positive")
    return sigma_squared


def nested_zentropy_three_state(
    centers: np.ndarray,
    sizes: np.ndarray,
    fine_to_coarse: np.ndarray,
    b_features: Sequence[int],
    sigma_squared: Optional[float] = None,
    relation_floor: float = RELATION_FLOOR,
    total_squared: Optional[np.ndarray] = None,
) -> NestedZResult:
    """Compute strict coarse->fine->B/T-support-state nested Zentropy."""
    center_array = np.asarray(centers, dtype=np.float64)
    size_array = np.asarray(sizes, dtype=np.int64).reshape(-1)
    mapping = np.asarray(fine_to_coarse, dtype=np.int64).reshape(-1)
    if center_array.ndim != 2 or center_array.shape[0] != size_array.size:
        raise ValueError("centers and sizes have incompatible shapes")
    if mapping.size != size_array.size or np.any(mapping < 0):
        raise ValueError("invalid fine-to-coarse mapping")
    if np.any(size_array <= 0) or int(size_array.sum()) < 2:
        raise ValueError("strict B/T support requires at least two total samples")
    _require_finite(center_array, "fine centers")
    feature_count = int(center_array.shape[1])
    b = np.asarray(sorted(set(int(value) for value in b_features)), dtype=np.int64)
    if b.size == 0 or np.any(b < 0) or np.any(b >= feature_count):
        raise ValueError("B must contain valid feature indices")
    b_set = set(int(value) for value in b)
    t = np.asarray(
        [value for value in range(feature_count) if value not in b_set],
        dtype=np.int64,
    )
    total = (
        _pairwise_total_squared(center_array)
        if total_squared is None
        else np.asarray(total_squared, dtype=np.float64)
    )
    if total.shape != (size_array.size, size_array.size):
        raise ValueError("total_squared has an invalid shape")
    b_difference = center_array[:, None, b] - center_array[None, :, b]
    b_sum = np.sum(b_difference * b_difference, axis=2)
    distance_b = b_sum / int(b.size)
    complement_degenerate = bool(t.size == 0)
    if t.size:
        t_sum = np.maximum(total - b_sum, 0.0)
        distance_t = t_sum / int(t.size)
    else:
        # Explicit one-feature/full-B degeneration: B and T coincide, so all
        # support belongs to the common state and relevance is necessarily zero.
        distance_t = distance_b.copy()
    sigma = (
        _bandwidth_squared(total, feature_count)
        if sigma_squared is None
        else float(sigma_squared)
    )
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_squared must be finite and positive")
    with np.errstate(under="ignore"):
        relation_b = np.exp((-0.5 * distance_b) / sigma)
        relation_t = np.exp((-0.5 * distance_t) / sigma)
    np.maximum(relation_b, float(relation_floor), out=relation_b)
    np.maximum(relation_t, float(relation_floor), out=relation_t)
    np.fill_diagonal(relation_b, 1.0)
    np.fill_diagonal(relation_t, 1.0)

    size_float = size_array.astype(np.float64)
    common_relation = np.minimum(relation_b, relation_t)
    np.fill_diagonal(common_relation, 0.0)
    common = common_relation @ size_float + (size_float - 1.0)
    difference = relation_b - relation_t
    b_only = np.maximum(difference, 0.0) @ size_float
    t_only = np.maximum(-difference, 0.0) @ size_float
    masses = np.column_stack((common, b_only, t_only))
    union = np.sum(masses, axis=1)
    if np.any(union <= 0.0) or not np.all(np.isfinite(union)):
        raise ValueError("zero or non-finite B/T union support")
    probabilities = masses / union[:, None]
    probability_sum_error = float(
        np.max(np.abs(np.sum(probabilities, axis=1) - 1.0))
    )
    probability_min = float(np.min(probabilities))
    probability_max = float(np.max(probabilities))
    if probability_sum_error > 1.0e-12 or probability_min < -1.0e-14:
        raise RuntimeError("invalid three-state probability distribution")

    weights = size_float / float(np.sum(size_float))
    local_entropy = _entropy(probabilities, axis=1)
    # The three states are themselves a strict two-level hierarchy:
    # common versus difference, followed by B-only versus T-only inside the
    # difference branch.  A zero difference mass contributes exactly zero.
    common_probability = probabilities[:, 0]
    difference_probability = probabilities[:, 1] + probabilities[:, 2]
    top_binary = _entropy(
        np.column_stack((common_probability, difference_probability)), axis=1
    )
    direction_probability = np.zeros_like(difference_probability)
    np.divide(
        probabilities[:, 1],
        difference_probability,
        out=direction_probability,
        where=difference_probability > 0.0,
    )
    direction_probability = np.clip(direction_probability, 0.0, 1.0)
    direction_binary = _entropy(
        np.column_stack((direction_probability, 1.0 - direction_probability)),
        axis=1,
    )
    support_recursive_entropy = (
        top_binary + difference_probability * direction_binary
    )
    support_recursive_error = float(
        np.max(np.abs(local_entropy - support_recursive_entropy))
    )
    if support_recursive_error > IDENTITY_TOL:
        raise RuntimeError("three-state support entropy recursion disagrees")
    information = -np.log2(weights) + local_entropy
    z_flat = float(_entropy(weights) + np.dot(weights, local_entropy))
    joint = weights[:, None] * probabilities
    z_joint = float(_entropy(joint.ravel()))

    coarse_count = int(np.max(mapping)) + 1
    coarse_weights = np.bincount(
        mapping, weights=weights, minlength=coarse_count
    ).astype(np.float64)
    if np.any(coarse_weights <= 0.0):
        raise RuntimeError("fine-to-coarse mapping contains an empty coarse group")
    per_coarse = np.empty(coarse_count, dtype=np.float64)
    for coarse in range(coarse_count):
        selected = mapping == coarse
        conditional = weights[selected] / coarse_weights[coarse]
        per_coarse[coarse] = float(
            _entropy(conditional) + np.dot(conditional, local_entropy[selected])
        )
    z_nested = float(_entropy(coarse_weights) + np.dot(coarse_weights, per_coarse))
    nested_error = max(abs(z_flat - z_nested), abs(z_flat - z_joint))
    if nested_error > IDENTITY_TOL:
        raise RuntimeError(
            "flat, nested, and direct joint Zentropy computations disagree"
        )

    marginal = np.dot(weights, probabilities)
    marginal_entropy = float(_entropy(marginal))
    conditional_entropy = float(np.dot(weights, local_entropy))
    mutual_information = marginal_entropy - conditional_entropy
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, marginal_entropy)
    if mutual_information < -tolerance:
        raise RuntimeError("computed negative mutual information beyond tolerance")
    if marginal_entropy <= tolerance or mutual_information <= tolerance:
        relevance = 0.0
    else:
        relevance = float(np.clip(mutual_information / marginal_entropy, 0.0, 1.0))
    return NestedZResult(
        probabilities=probabilities,
        local_entropy=np.asarray(local_entropy, dtype=np.float64),
        information=np.asarray(information, dtype=np.float64),
        z_flat=z_flat,
        z_nested=z_nested,
        z_joint=z_joint,
        nested_error=float(nested_error),
        marginal_probabilities=np.asarray(marginal, dtype=np.float64),
        marginal_entropy=marginal_entropy,
        conditional_entropy=conditional_entropy,
        relevance=relevance,
        support_recursive_entropy=np.asarray(
            support_recursive_entropy, dtype=np.float64
        ),
        support_recursive_error=support_recursive_error,
        probability_sum_max_error=probability_sum_error,
        probability_min=probability_min,
        probability_max=probability_max,
        b_features=b,
        t_features=t,
        complement_degenerate=complement_degenerate,
    )


def _discretize_features(
    x: np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    sample_count, feature_count = x.shape
    default_bins = int(math.ceil(math.log2(max(sample_count, 2)) + 1.0))
    codes: List[np.ndarray] = []
    entropies = np.zeros(feature_count, dtype=np.float64)
    contrasts = np.zeros(feature_count, dtype=np.float64)
    for feature in range(feature_count):
        column = x[:, feature]
        unique = np.unique(column)
        bin_count = min(int(unique.size), default_bins)
        if bin_count <= 1:
            code = np.zeros(sample_count, dtype=np.int32)
            codes.append(code)
            continue
        if unique.size <= default_bins:
            _, code = np.unique(column, return_inverse=True)
            code = code.astype(np.int32, copy=False)
            bin_count = int(code.max()) + 1
        else:
            edges = np.linspace(float(column.min()), float(column.max()), bin_count + 1)
            code = np.searchsorted(edges[1:-1], column, side="right").astype(np.int32)
        counts = np.bincount(code, minlength=bin_count).astype(np.float64)
        probabilities = counts / float(sample_count)
        surprise = -np.log(probabilities[code] + EPS)
        entropy = float(np.mean(surprise))
        spread = math.sqrt(max(float(np.var(surprise)), 0.0))
        codes.append(code)
        entropies[feature] = entropy
        contrasts[feature] = spread / (entropy + spread + EPS)
    return codes, entropies, contrasts


def _normalized_mi_matrix(
    codes: Sequence[np.ndarray], entropies: np.ndarray
) -> np.ndarray:
    feature_count = len(codes)
    sample_count = len(codes[0]) if feature_count else 0
    nmi = np.eye(feature_count, dtype=np.float64)
    bin_counts = [int(code.max()) + 1 for code in codes]
    marginal_counts = [
        np.bincount(codes[j], minlength=bin_counts[j]).astype(np.float64)
        for j in range(feature_count)
    ]
    for first in range(feature_count):
        if entropies[first] <= EPS:
            nmi[first, first] = 0.0
        first_code = codes[first]
        first_bins = bin_counts[first]
        first_count = marginal_counts[first]
        for second in range(first + 1, feature_count):
            if entropies[first] <= EPS or entropies[second] <= EPS:
                value = 0.0
            else:
                second_code = codes[second]
                second_bins = bin_counts[second]
                joint = np.bincount(
                    first_code * second_bins + second_code,
                    minlength=first_bins * second_bins,
                ).reshape(first_bins, second_bins)
                rows, columns = np.nonzero(joint)
                joint_counts = joint[rows, columns].astype(np.float64)
                ratio = (joint_counts * sample_count) / (
                    first_count[rows] * marginal_counts[second][columns] + EPS
                )
                mutual_information = float(
                    np.sum((joint_counts / sample_count) * np.log(ratio + EPS))
                )
                value = mutual_information / (
                    math.sqrt(entropies[first] * entropies[second]) + EPS
                )
                value = float(np.clip(value, 0.0, 1.0))
            nmi[first, second] = value
            nmi[second, first] = value
    return nmi


def _strict_feature_path(
    relevance: np.ndarray, nmi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Build one deterministic path; every rho model is a prefix of it."""
    feature_count = int(relevance.size)
    selected: List[int] = []
    available = np.ones(feature_count, dtype=bool)
    relevance_sum = 0.0
    redundancy_sum = 0.0
    redundancy_with_selected = np.zeros(feature_count, dtype=np.float64)
    utilities = np.empty(feature_count, dtype=np.float64)
    for step in range(feature_count):
        candidates = np.flatnonzero(available)
        new_size = step + 1
        if step == 0:
            values = relevance[candidates]
        else:
            # Incremental pairwise redundancy makes the complete path O(d^2)
            # instead of repeatedly summing growing candidate blocks (O(d^3)).
            redundancy_add = redundancy_with_selected[candidates]
            pair_count = new_size * (new_size - 1) / 2.0
            mean_relevance = (relevance_sum + relevance[candidates]) / new_size
            mean_redundancy = (redundancy_sum + redundancy_add) / pair_count
            values = mean_relevance * (1.0 - mean_redundancy)
        best_value = float(np.max(values))
        tied = candidates[np.isclose(values, best_value, rtol=1.0e-14, atol=1.0e-15)]
        best = int(np.min(tied))
        if selected:
            redundancy_sum += float(nmi[best, np.asarray(selected, dtype=np.int64)].sum())
        relevance_sum += float(relevance[best])
        selected.append(best)
        available[best] = False
        redundancy_with_selected += nmi[:, best]
        utilities[step] = best_value
    return np.asarray(selected, dtype=np.int64), utilities


def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(int(base_seed))] + [str(part) for part in parts])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _generate_subspaces(
    feature_count: int, requested_count: int, rng: np.random.Generator
) -> List[np.ndarray]:
    if feature_count <= 1:
        return [np.asarray([0], dtype=np.int64)]
    size = min(
        feature_count - 1,
        max(1, int(math.ceil(math.sqrt(feature_count)))),
    )
    combination_count = math.comb(feature_count, size)
    effective_count = min(int(requested_count), combination_count)
    if combination_count <= requested_count:
        return [
            np.asarray(combination, dtype=np.int64)
            for combination in itertools.combinations(range(feature_count), size)
        ]
    seen: Set[Tuple[int, ...]] = set()
    subspaces: List[np.ndarray] = []
    while len(subspaces) < effective_count:
        subset = tuple(
            sorted(rng.choice(feature_count, size=size, replace=False).tolist())
        )
        if subset not in seen:
            seen.add(subset)
            subspaces.append(np.asarray(subset, dtype=np.int64))
    return subspaces


def _prepare_subspace_caches(
    x_selected: np.ndarray,
    hierarchy: BallHierarchy,
    subspaces: Sequence[np.ndarray],
    relation_floor: float,
) -> Tuple[List[SubspaceCache], float]:
    sample_count = int(x_selected.shape[0])
    fine_count = int(hierarchy.fine_sizes.size)
    centers_full = hierarchy.fine_centers
    total_squared = _pairwise_total_squared(centers_full)
    sigma_squared = _bandwidth_squared(total_squared, x_selected.shape[1])
    caches: List[SubspaceCache] = []
    for local_features in subspaces:
        sample_view = x_selected[:, local_features]
        centers = centers_full[:, local_features]
        radii = np.zeros(fine_count, dtype=np.float64)
        within_distance = np.zeros(sample_count, dtype=np.float64)
        for fine, node_value in enumerate(hierarchy.fine_node_ids):
            indices = hierarchy.nodes[int(node_value)].indices
            difference = sample_view[indices] - centers[fine]
            distance = np.sqrt(np.mean(difference * difference, axis=1))
            within_distance[indices] = distance
            radii[fine] = (
                math.sqrt(float(np.mean(distance * distance)))
                if distance.size
                else 0.0
            )
        positive_radii = radii[radii > EPS]
        median_radius = float(np.median(positive_radii)) if positive_radii.size else 1.0
        center_difference = centers[:, None, :] - centers[None, :, :]
        center_distance = np.sqrt(np.mean(center_difference * center_difference, axis=2))
        denominator = radii[:, None] + radii[None, :] + median_radius + EPS
        distances = center_distance / denominator
        np.fill_diagonal(distances, np.inf)
        neighbor_order = np.argsort(distances, axis=1, kind="mergesort")
        assigned_radius = radii[hierarchy.fine_membership]
        within_deviation = within_distance / (
            within_distance + assigned_radius + EPS
        )
        nested = nested_zentropy_three_state(
            centers=centers_full,
            sizes=hierarchy.fine_sizes,
            fine_to_coarse=hierarchy.fine_to_coarse,
            b_features=local_features,
            sigma_squared=sigma_squared,
            relation_floor=relation_floor,
            total_squared=total_squared,
        )
        complement = nested.t_features
        caches.append(
            SubspaceCache(
                local_features=np.asarray(local_features, dtype=np.int64),
                complement_features=np.asarray(complement, dtype=np.int64),
                nested=nested,
                centers=centers,
                radii=radii,
                distances=distances,
                neighbor_order=neighbor_order,
                within_deviation=within_deviation,
            )
        )
    return caches, sigma_squared


def _weighted_midrank(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    result = np.empty_like(values, dtype=np.float64)
    total = float(np.sum(weights))
    cumulative = 0.0
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and np.isclose(
            sorted_values[end],
            sorted_values[start],
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            end += 1
        group_indices = order[start:end]
        group_weight = float(np.sum(weights[group_indices]))
        result[group_indices] = (cumulative + 0.5 * group_weight) / max(total, EPS)
        cumulative += group_weight
        start = end
    return result


def _subspace_scores_for_k(
    caches: Sequence[SubspaceCache],
    ball_sizes: np.ndarray,
    membership: np.ndarray,
    requested_k: int,
) -> Tuple[np.ndarray, int]:
    fine_count = int(ball_sizes.size)
    if fine_count < 2:
        return np.vstack([cache.within_deviation for cache in caches]), 0
    effective_k = min(max(1, int(requested_k)), fine_count - 1)
    sample_count = int(membership.size)
    rarity = np.log((sample_count + fine_count) / (ball_sizes + 1.0)) / math.log(
        sample_count + fine_count
    )
    rows: List[np.ndarray] = []
    row_indices = np.arange(fine_count)[:, None]
    for cache in caches:
        neighbors = cache.neighbor_order[:, :effective_k]
        neighbor_mass = ball_sizes[neighbors]
        local_distances = cache.distances[row_indices, neighbors]
        denominator = np.sum(neighbor_mass, axis=1)
        local_scale = np.sum(neighbor_mass * local_distances, axis=1) / denominator
        neighbor_scale = local_scale[neighbors]
        relative_distance = local_distances / (neighbor_scale + EPS)
        ball_score = (1.0 + rarity) * (
            np.sum(neighbor_mass * relative_distance, axis=1) / denominator
        )
        ball_rank = _weighted_midrank(ball_score, ball_sizes)
        sample_score = ball_rank[membership] * (1.0 + cache.within_deviation)
        rows.append(sample_score)
    return np.vstack(rows), effective_k


def _fuse_subspace_prefix(
    subspace_scores: np.ndarray,
    caches: Sequence[SubspaceCache],
    requested_count: int,
) -> Tuple[np.ndarray, int, np.ndarray, bool]:
    effective_count = min(max(1, int(requested_count)), len(caches))
    relevance = np.asarray(
        [cache.nested.relevance for cache in caches[:effective_count]],
        dtype=np.float64,
    )
    total = float(np.sum(relevance))
    tolerance = 64.0 * np.finfo(np.float64).eps
    if total <= tolerance:
        alpha = np.full(effective_count, 1.0 / effective_count, dtype=np.float64)
        uniform_fallback = True
    else:
        alpha = relevance / total
        uniform_fallback = False
    score = np.sum(subspace_scores[:effective_count] * alpha[:, None], axis=0)
    _require_finite(score, "fused anomaly scores")
    return score, effective_count, alpha, uniform_fallback


def _prefix_checks(
    feature_order: np.ndarray,
    retain_counts: Sequence[int],
    artifacts: Dict[int, RhoArtifact],
    requested_m: Sequence[int],
    requested_k: Sequence[int],
) -> Tuple[bool, bool, bool]:
    feature_ok = True
    previous: Optional[np.ndarray] = None
    for count in sorted(set(int(value) for value in retain_counts)):
        if count not in artifacts:
            feature_ok = False
            continue
        current = np.asarray(artifacts[count].selected_local, dtype=np.int64)
        if not np.array_equal(current, feature_order[:count]):
            feature_ok = False
        if previous is not None and not np.array_equal(previous, current[: previous.size]):
            feature_ok = False
        previous = current
    subspace_ok = True
    neighbor_ok = True
    for artifact in artifacts.values():
        previous_subspaces: Optional[List[np.ndarray]] = None
        for value in sorted(set(int(item) for item in requested_m)):
            effective = min(value, len(artifact.subspaces))
            current_subspaces = [
                np.asarray(cache.local_features, dtype=np.int64)
                for cache in artifact.caches[:effective]
            ]
            if len(current_subspaces) != effective or any(
                not np.array_equal(current_subspaces[index], artifact.subspaces[index])
                for index in range(effective)
            ):
                subspace_ok = False
            if previous_subspaces is not None and (
                len(previous_subspaces) > len(current_subspaces)
                or any(
                    not np.array_equal(previous_subspaces[index], current_subspaces[index])
                    for index in range(len(previous_subspaces))
                )
            ):
                subspace_ok = False
            previous_subspaces = current_subspaces
        fine_count = int(artifact.hierarchy.fine_sizes.size)
        if fine_count < 2:
            continue
        effective_values = sorted(
            set(min(max(1, int(value)), fine_count - 1) for value in requested_k)
        )
        for cache in artifact.caches:
            previous_neighbors: Optional[np.ndarray] = None
            for effective in effective_values:
                current_neighbors = cache.neighbor_order[:, :effective]
                if previous_neighbors is not None and not np.array_equal(
                    previous_neighbors,
                    current_neighbors[:, : previous_neighbors.shape[1]],
                ):
                    neighbor_ok = False
                previous_neighbors = current_neighbors
    return feature_ok, subspace_ok, neighbor_ok


def compute_grid_scores(
    x_raw: np.ndarray,
    rho_grid: Sequence[float],
    subspace_grid: Sequence[int],
    k_grid: Sequence[int],
    config: Optional[ModelConfig] = None,
    dataset_name: str = "dataset",
) -> Dict[str, Any]:
    """Compute every score vector without accepting or inspecting labels."""
    if config is None:
        config = ModelConfig()
    if not rho_grid or not subspace_grid or not k_grid:
        raise ValueError("all parameter grids must be non-empty")
    rho_values = sorted(set(float(value) for value in rho_grid))
    subspace_values = sorted(set(int(value) for value in subspace_grid))
    k_values = sorted(set(int(value) for value in k_grid))
    if any(value <= 0.0 or value > 1.0 for value in rho_values):
        raise ValueError("rho values must be in (0, 1]")
    if any(value < 1 for value in subspace_values + k_values):
        raise ValueError("M and k values must be positive")

    started = time.perf_counter()
    x, preprocessing = preprocess_features(x_raw)
    sample_count, active_features = x.shape
    reference_hierarchy = build_hierarchical_balls(x, config.max_balls)
    reference_total = _pairwise_total_squared(reference_hierarchy.fine_centers)
    reference_sigma = _bandwidth_squared(reference_total, active_features)
    feature_relevance = np.empty(active_features, dtype=np.float64)
    feature_z_flat = np.empty(active_features, dtype=np.float64)
    feature_z_nested = np.empty(active_features, dtype=np.float64)
    feature_nested_error = np.empty(active_features, dtype=np.float64)
    maximum_probability_error = 0.0
    minimum_probability = 1.0
    maximum_probability = 0.0
    maximum_support_recursive_error = 0.0
    for feature in range(active_features):
        result = nested_zentropy_three_state(
            centers=reference_hierarchy.fine_centers,
            sizes=reference_hierarchy.fine_sizes,
            fine_to_coarse=reference_hierarchy.fine_to_coarse,
            b_features=[feature],
            sigma_squared=reference_sigma,
            relation_floor=config.relation_floor,
            total_squared=reference_total,
        )
        feature_relevance[feature] = result.relevance
        feature_z_flat[feature] = result.z_flat
        feature_z_nested[feature] = result.z_nested
        feature_nested_error[feature] = result.nested_error
        maximum_probability_error = max(
            maximum_probability_error, result.probability_sum_max_error
        )
        minimum_probability = min(minimum_probability, result.probability_min)
        maximum_probability = max(maximum_probability, result.probability_max)
        maximum_support_recursive_error = max(
            maximum_support_recursive_error, result.support_recursive_error
        )

    codes, feature_entropies, legacy_contrasts = _discretize_features(x)
    feature_nmi = _normalized_mi_matrix(codes, feature_entropies)
    relevance_tolerance = 64.0 * np.finfo(np.float64).eps
    if float(np.sum(feature_relevance)) > relevance_tolerance:
        selection_relevance = feature_relevance.copy()
        feature_selection_fallback_mode = "nested_relevance"
    elif float(np.sum(legacy_contrasts)) > relevance_tolerance:
        selection_relevance = legacy_contrasts.copy()
        feature_selection_fallback_mode = "legacy_contrast"
    else:
        # Uniform relevance leaves the later greedy steps free to prefer lower
        # redundancy instead of ignoring NMI after an arbitrary first tie.
        selection_relevance = np.ones(active_features, dtype=np.float64)
        feature_selection_fallback_mode = "uniform"
    feature_order, path_utilities = _strict_feature_path(
        selection_relevance, feature_nmi
    )
    retain_counts = [
        min(
            active_features,
            max(
                2 if active_features > 1 else 1,
                int(math.ceil(rho * active_features)),
            ),
        )
        for rho in rho_values
    ]

    artifacts: Dict[int, RhoArtifact] = {}
    for selected_count in sorted(set(retain_counts)):
        selected_local = feature_order[:selected_count].copy()
        x_selected = x[:, selected_local]
        hierarchy = build_hierarchical_balls(x_selected, config.max_balls)
        rng = np.random.default_rng(
            _stable_seed(config.seed, dataset_name, "subspaces", selected_count)
        )
        subspaces = _generate_subspaces(
            selected_count, max(subspace_values), rng
        )
        caches, sigma_squared = _prepare_subspace_caches(
            x_selected=x_selected,
            hierarchy=hierarchy,
            subspaces=subspaces,
            relation_floor=config.relation_floor,
        )
        for cache in caches:
            maximum_probability_error = max(
                maximum_probability_error,
                cache.nested.probability_sum_max_error,
            )
            minimum_probability = min(minimum_probability, cache.nested.probability_min)
            maximum_probability = max(maximum_probability, cache.nested.probability_max)
            maximum_support_recursive_error = max(
                maximum_support_recursive_error,
                cache.nested.support_recursive_error,
            )
        artifacts[selected_count] = RhoArtifact(
            selected_count=selected_count,
            selected_local=selected_local,
            selected_original=preprocessing.original_feature_indices[selected_local],
            hierarchy=hierarchy,
            subspaces=subspaces,
            caches=caches,
            sigma_squared=sigma_squared,
            path_utility=float(path_utilities[selected_count - 1]),
        )

    score_rows: List[np.ndarray] = []
    detail_rows: List[Dict[str, Any]] = []
    for rho, selected_count in zip(rho_values, retain_counts):
        artifact = artifacts[selected_count]
        component_cache: Dict[int, Tuple[np.ndarray, int]] = {}
        for requested_k in k_values:
            component_scores, effective_k = _subspace_scores_for_k(
                artifact.caches,
                artifact.hierarchy.fine_sizes.astype(np.float64),
                artifact.hierarchy.fine_membership,
                requested_k,
            )
            component_cache[requested_k] = (component_scores, effective_k)
            for requested_m in subspace_values:
                score, effective_m, alpha, uniform_fallback = _fuse_subspace_prefix(
                    component_scores,
                    artifact.caches,
                    requested_m,
                )
                config_index = len(score_rows)
                score_rows.append(score)
                prefix_nested = [
                    cache.nested for cache in artifact.caches[:effective_m]
                ]
                detail_rows.append(
                    {
                        "dataset": dataset_name,
                        "config_index": config_index,
                        "rho": float(rho),
                        "requested_subspaces": int(requested_m),
                        "effective_subspaces": int(effective_m),
                        "requested_k": int(requested_k),
                        "effective_k": int(effective_k),
                        "selected_features": int(selected_count),
                        "n_fine_balls": int(artifact.hierarchy.fine_sizes.size),
                        "n_coarse_balls": int(artifact.hierarchy.coarse_sizes.size),
                        "path_utility": float(artifact.path_utility),
                        "mean_subspace_relevance": float(
                            np.mean([item.relevance for item in prefix_nested])
                        ),
                        "mean_z_flat": float(
                            np.mean([item.z_flat for item in prefix_nested])
                        ),
                        "mean_z_nested": float(
                            np.mean([item.z_nested for item in prefix_nested])
                        ),
                        "nested_entropy_max_error": float(
                            max(item.nested_error for item in prefix_nested)
                        ),
                        "support_recursive_max_error": float(
                            max(item.support_recursive_error for item in prefix_nested)
                        ),
                        "uniform_fusion_fallback": bool(uniform_fallback),
                        "fusion_weight_sum": float(np.sum(alpha)),
                        "sigma_squared": float(artifact.sigma_squared),
                    }
                )

    scores = np.vstack(score_rows)
    feature_prefix_ok, subspace_prefix_ok, neighbor_prefix_ok = _prefix_checks(
        feature_order,
        retain_counts,
        artifacts,
        subspace_values,
        k_values,
    )
    nested_max = max(
        float(np.max(feature_nested_error)) if feature_nested_error.size else 0.0,
        max(
            (
                cache.nested.nested_error
                for artifact in artifacts.values()
                for cache in artifact.caches
            ),
            default=0.0,
        ),
    )
    if nested_max > IDENTITY_TOL:
        raise RuntimeError("nested entropy audit exceeded tolerance")
    return {
        "scores": scores,
        "details": detail_rows,
        "preprocessing": preprocessing,
        "reference_hierarchy": reference_hierarchy,
        "reference_sigma_squared": reference_sigma,
        "feature_relevance": feature_relevance,
        "feature_selection_relevance": selection_relevance,
        "feature_selection_fallback_mode": feature_selection_fallback_mode,
        "feature_selection_uniform_or_contrast_fallback": bool(
            feature_selection_fallback_mode != "nested_relevance"
        ),
        "feature_z_flat": feature_z_flat,
        "feature_z_nested": feature_z_nested,
        "feature_nested_error": feature_nested_error,
        "feature_entropy": feature_entropies,
        "legacy_contrast_diagnostic": legacy_contrasts,
        "feature_nmi": feature_nmi,
        "feature_order": feature_order,
        "path_utilities": path_utilities,
        "artifacts": artifacts,
        "retain_counts": np.asarray(retain_counts, dtype=np.int64),
        "nested_entropy_max_error": float(nested_max),
        "probability_sum_max_error": float(maximum_probability_error),
        "probability_min": float(minimum_probability),
        "probability_max": float(maximum_probability),
        "support_recursive_max_error": float(maximum_support_recursive_error),
        "feature_prefix_ok": bool(feature_prefix_ok),
        "subspace_prefix_ok": bool(subspace_prefix_ok),
        "neighbor_prefix_ok": bool(neighbor_prefix_ok),
        "model_seconds": float(time.perf_counter() - started),
        "n_samples": int(sample_count),
        "n_active_features": int(active_features),
    }


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        loaded = loadmat(str(path))
        if "trandata" not in loaded:
            raise ValueError("missing 'trandata' matrix")
        matrix = np.asarray(loaded["trandata"])
    except NotImplementedError as exc:
        try:
            import h5py  # type: ignore
        except ImportError:
            raise ValueError(
                "MATLAB v7.3 input requires the optional h5py package"
            ) from exc
        with h5py.File(str(path), "r") as handle:
            if "trandata" not in handle:
                raise ValueError("missing 'trandata' matrix")
            matrix = np.asarray(handle["trandata"]).T
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("invalid trandata shape: %r" % (matrix.shape,))
    if np.iscomplexobj(matrix) or not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("trandata must be a real numeric matrix")
    x = np.asarray(matrix[:, :-1], dtype=np.float64)
    y = np.asarray(matrix[:, -1]).reshape(-1)
    _require_finite(x, "features")
    _require_finite(np.asarray(y, dtype=np.float64), "labels")
    unique = np.unique(y)
    if not np.array_equal(unique, np.asarray([0, 1])) and not np.array_equal(
        unique, np.asarray([0.0, 1.0])
    ):
        raise ValueError("expected both binary labels 0/1, found %r" % unique.tolist())
    return x, y.astype(np.int8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _tree_npz(prefix: str, hierarchy: BallHierarchy) -> Dict[str, np.ndarray]:
    return {
        prefix + "tree_parent": np.asarray(
            [node.parent for node in hierarchy.nodes], dtype=np.int64
        ),
        prefix + "tree_depth": np.asarray(
            [node.depth for node in hierarchy.nodes], dtype=np.int64
        ),
        prefix + "tree_size": np.asarray(
            [node.indices.size for node in hierarchy.nodes], dtype=np.int64
        ),
        prefix + "tree_centers": np.vstack(
            [node.center for node in hierarchy.nodes]
        ),
        prefix + "tree_sse": np.asarray(
            [node.sse for node in hierarchy.nodes], dtype=np.float64
        ),
        prefix + "tree_left": np.asarray(
            [node.left for node in hierarchy.nodes], dtype=np.int64
        ),
        prefix + "tree_right": np.asarray(
            [node.right for node in hierarchy.nodes], dtype=np.int64
        ),
        prefix + "accepted_splits": hierarchy.accepted_splits,
        prefix + "fine_node_ids": hierarchy.fine_node_ids,
        prefix + "coarse_node_ids": hierarchy.coarse_node_ids,
        prefix + "fine_membership": hierarchy.fine_membership,
        prefix + "coarse_membership": hierarchy.coarse_membership,
        prefix + "fine_to_coarse": hierarchy.fine_to_coarse,
        prefix + "fine_centers": hierarchy.fine_centers,
        prefix + "fine_sizes": hierarchy.fine_sizes,
        prefix + "coarse_sizes": hierarchy.coarse_sizes,
    }


def _best_key(row: Dict[str, Any], metric: str) -> Tuple[float, int, int, float]:
    return (
        float(row[metric]),
        -int(row["requested_subspaces"]),
        -int(row["requested_k"]),
        -float(row["rho"]),
    )


def _artifact_npz(
    prefix: str,
    artifact: RhoArtifact,
    effective_subspaces: int,
) -> Dict[str, np.ndarray]:
    output = _tree_npz(prefix, artifact.hierarchy)
    caches = artifact.caches[:effective_subspaces]
    relevance = np.asarray(
        [cache.nested.relevance for cache in caches], dtype=np.float64
    )
    fusion_weights_by_prefix = np.zeros(
        (effective_subspaces, effective_subspaces), dtype=np.float64
    )
    fusion_uniform_fallback = np.zeros(effective_subspaces, dtype=bool)
    relevance_tolerance = 64.0 * np.finfo(np.float64).eps
    for count in range(1, effective_subspaces + 1):
        total = float(np.sum(relevance[:count]))
        if total <= relevance_tolerance:
            fusion_weights_by_prefix[count - 1, :count] = 1.0 / count
            fusion_uniform_fallback[count - 1] = True
        else:
            fusion_weights_by_prefix[count - 1, :count] = relevance[:count] / total
    output.update(
        {
            prefix + "selected_local_features": artifact.selected_local,
            prefix + "selected_original_features": artifact.selected_original,
            prefix + "subspaces": np.vstack(
                [cache.local_features for cache in caches]
            ),
            prefix + "state_probabilities": np.stack(
                [cache.nested.probabilities for cache in caches], axis=0
            ),
            prefix + "subspace_relevance": relevance,
            # Row m-1 contains the exact alpha weights for the first m
            # subspaces.  This makes every M-prefix fusion independently
            # reproducible without consulting implementation code.
            prefix + "fusion_weights_by_prefix": fusion_weights_by_prefix,
            prefix + "fusion_uniform_fallback_by_prefix": (
                fusion_uniform_fallback
            ),
            prefix + "fusion_weights": fusion_weights_by_prefix[
                effective_subspaces - 1
            ],
            prefix + "neighbor_order": np.stack(
                [cache.neighbor_order.astype(np.int32) for cache in caches],
                axis=0,
            ),
            prefix + "z_flat": np.asarray(
                [cache.nested.z_flat for cache in caches], dtype=np.float64
            ),
            prefix + "z_nested": np.asarray(
                [cache.nested.z_nested for cache in caches], dtype=np.float64
            ),
            prefix + "z_joint": np.asarray(
                [cache.nested.z_joint for cache in caches], dtype=np.float64
            ),
            prefix + "nested_error": np.asarray(
                [cache.nested.nested_error for cache in caches], dtype=np.float64
            ),
            prefix + "support_recursive_entropy": np.stack(
                [cache.nested.support_recursive_entropy for cache in caches],
                axis=0,
            ),
            prefix + "support_recursive_error": np.asarray(
                [cache.nested.support_recursive_error for cache in caches],
                dtype=np.float64,
            ),
            prefix + "sigma_squared": np.asarray(artifact.sigma_squared),
        }
    )
    return output


def _atomic_npz(path: Path, arrays: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(str(temporary), str(path))


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(str(temporary), str(path))


def _atomic_csv(
    path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


DETAIL_FIELDS = [
    "dataset",
    "config_index",
    "rho",
    "requested_subspaces",
    "effective_subspaces",
    "requested_k",
    "effective_k",
    "auc",
    "ap",
    "selected_features",
    "n_fine_balls",
    "n_coarse_balls",
    "path_utility",
    "mean_subspace_relevance",
    "mean_z_flat",
    "mean_z_nested",
    "nested_entropy_max_error",
    "support_recursive_max_error",
    "uniform_fusion_fallback",
    "fusion_weight_sum",
    "sigma_squared",
    "input_sha256",
]


BEST_FIELDS = [
    "dataset",
    "status",
    "error",
    "n_samples",
    "n_original_features",
    "n_active_features",
    "n_anomalies",
    "anomaly_rate",
    "selection_metric",
    "best_metric_value",
    "auc",
    "ap",
    "rho",
    "requested_subspaces",
    "effective_subspaces",
    "requested_k",
    "effective_k",
    "selected_features",
    "n_fine_balls",
    "n_coarse_balls",
    "nested_entropy_max_error",
    "model_seconds",
    "metrics_seconds",
    "total_seconds",
    "input_sha256",
    "result_npz",
]


DATASET_SUMMARY_FIELDS = [
    "dataset",
    "status",
    "error",
    "n_samples",
    "n_original_features",
    "n_active_features",
    "n_anomalies",
    "anomaly_rate",
    "best_auc",
    "ap_at_best_auc",
    "best_ap",
    "auc_at_best_ap",
    "nested_entropy_max_error",
    "input_sha256",
    "result_npz",
]


MANIFEST_FIELDS = [
    "dataset",
    "input_path",
    "input_sha256",
    "file_size_bytes",
]


def evaluate_dataset_file(
    path: Path,
    output_dir: Path,
    rho_grid: Sequence[float],
    subspace_grid: Sequence[int],
    k_grid: Sequence[int],
    config: ModelConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    dataset_started = time.perf_counter()
    input_hash = _sha256(path)
    x, y = load_dataset(path)
    grid = compute_grid_scores(
        x,
        rho_grid=rho_grid,
        subspace_grid=subspace_grid,
        k_grid=k_grid,
        config=config,
        dataset_name=path.stem,
    )
    metric_started = time.perf_counter()
    auc_values = np.asarray(
        [roc_auc_score(y, score) for score in grid["scores"]], dtype=np.float64
    )
    ap_values = np.asarray(
        [average_precision_score(y, score) for score in grid["scores"]],
        dtype=np.float64,
    )
    metrics_seconds = time.perf_counter() - metric_started
    detail_rows: List[Dict[str, Any]] = []
    for row, auc, ap in zip(grid["details"], auc_values, ap_values):
        copied = dict(row)
        copied["auc"] = float(auc)
        copied["ap"] = float(ap)
        copied["input_sha256"] = input_hash
        detail_rows.append(copied)
    best_auc = max(detail_rows, key=lambda row: _best_key(row, "auc"))
    best_ap = max(detail_rows, key=lambda row: _best_key(row, "ap"))
    best_auc_index = int(best_auc["config_index"])
    best_ap_index = int(best_ap["config_index"])

    preprocessing = grid["preprocessing"]
    npz_arrays: Dict[str, Any] = {
        "labels": y,
        "scores": grid["scores"],
        "rho": np.asarray([row["rho"] for row in detail_rows], dtype=np.float64),
        "requested_subspaces": np.asarray(
            [row["requested_subspaces"] for row in detail_rows], dtype=np.int64
        ),
        "effective_subspaces": np.asarray(
            [row["effective_subspaces"] for row in detail_rows], dtype=np.int64
        ),
        "requested_k": np.asarray(
            [row["requested_k"] for row in detail_rows], dtype=np.int64
        ),
        "effective_k": np.asarray(
            [row["effective_k"] for row in detail_rows], dtype=np.int64
        ),
        "auc": auc_values,
        "ap": ap_values,
        "input_sha256": np.asarray(input_hash),
        "scores_by_config": grid["scores"],
        "config_rho": np.asarray([row["rho"] for row in detail_rows]),
        "config_M": np.asarray(
            [row["requested_subspaces"] for row in detail_rows], dtype=np.int64
        ),
        "config_k": np.asarray(
            [row["requested_k"] for row in detail_rows], dtype=np.int64
        ),
        "selected_feature_count": np.asarray(
            [row["selected_features"] for row in detail_rows], dtype=np.int64
        ),
        "n_fine_balls": np.asarray(
            [row["n_fine_balls"] for row in detail_rows], dtype=np.int64
        ),
        "n_coarse_balls": np.asarray(
            [row["n_coarse_balls"] for row in detail_rows], dtype=np.int64
        ),
        "original_feature_indices": preprocessing.original_feature_indices,
        "dropped_feature_indices": preprocessing.dropped_feature_indices,
        "feature_medians": preprocessing.medians,
        "feature_scales": preprocessing.scales,
        "feature_order_local": grid["feature_order"],
        "feature_order_original": preprocessing.original_feature_indices[
            grid["feature_order"]
        ],
        "feature_relevance": grid["feature_relevance"],
        "feature_selection_relevance": grid["feature_selection_relevance"],
        "feature_selection_uniform_or_contrast_fallback": np.asarray(
            grid["feature_selection_uniform_or_contrast_fallback"]
        ),
        "feature_selection_fallback_mode": np.asarray(
            grid["feature_selection_fallback_mode"]
        ),
        "feature_z_flat": grid["feature_z_flat"],
        "feature_z_nested": grid["feature_z_nested"],
        "feature_nested_error": grid["feature_nested_error"],
        "feature_nmi": grid["feature_nmi"],
        "path_utilities": grid["path_utilities"],
        "best_auc_config_index": np.asarray(best_auc_index, dtype=np.int64),
        "best_ap_config_index": np.asarray(best_ap_index, dtype=np.int64),
        "best_auc_scores": grid["scores"][best_auc_index],
        "best_ap_scores": grid["scores"][best_ap_index],
    }
    npz_arrays.update(_tree_npz("reference_", grid["reference_hierarchy"]))
    # Preserve every distinct rho-prefix model, not only the two oracle-best
    # configurations, so all grid score vectors retain auditable tree/state
    # evidence. Prefixes are keyed by selected feature count because multiple
    # rho values can map to the same count on low-dimensional datasets.
    for selected_count, artifact in sorted(grid["artifacts"].items()):
        npz_arrays.update(
            _artifact_npz(
                "selected_%d_" % int(selected_count),
                artifact,
                len(artifact.caches),
            )
        )
    best_auc_artifact = grid["artifacts"][int(best_auc["selected_features"])]
    best_ap_artifact = grid["artifacts"][int(best_ap["selected_features"])]
    npz_arrays.update(
        _artifact_npz(
            "best_auc_",
            best_auc_artifact,
            int(best_auc["effective_subspaces"]),
        )
    )
    npz_arrays.update(
        _artifact_npz(
            "best_ap_",
            best_ap_artifact,
            int(best_ap["effective_subspaces"]),
        )
    )
    result_path = output_dir / "per_dataset" / (path.stem + ".npz")
    _atomic_npz(result_path, npz_arrays)

    diagnostics = {
        "dataset": path.stem,
        "nested_entropy_max_error": grid["nested_entropy_max_error"],
        "probability_sum_max_error": grid["probability_sum_max_error"],
        "probability_min": grid["probability_min"],
        "probability_max": grid["probability_max"],
        "support_recursive_max_error": grid["support_recursive_max_error"],
        "feature_prefix_ok": grid["feature_prefix_ok"],
        "subspace_prefix_ok": grid["subspace_prefix_ok"],
        "neighbor_prefix_ok": grid["neighbor_prefix_ok"],
        "input_sha256": input_hash,
        "score_direction": "higher_is_anomaly",
        "label_informed_score_flip": False,
        "labels_used_by_model": False,
        "best_selection_protocol": "oracle_same_dataset_diagnostic_only",
        "n_configurations": int(len(detail_rows)),
        "n_samples": int(x.shape[0]),
        "n_original_features": int(x.shape[1]),
        "n_active_features": int(grid["n_active_features"]),
        "feature_order_original": preprocessing.original_feature_indices[
            grid["feature_order"]
        ].tolist(),
        "feature_selection_uniform_or_contrast_fallback": grid[
            "feature_selection_uniform_or_contrast_fallback"
        ],
        "feature_selection_fallback_mode": grid[
            "feature_selection_fallback_mode"
        ],
    }
    diagnostic_path = output_dir / "per_dataset" / (
        path.stem + "_diagnostics.json"
    )
    _atomic_json(diagnostic_path, diagnostics)

    total_seconds = time.perf_counter() - dataset_started

    def make_best_summary(row: Dict[str, Any], metric: str) -> Dict[str, Any]:
        return {
            "dataset": path.stem,
            "status": "ok",
            "error": "",
            "n_samples": int(x.shape[0]),
            "n_original_features": int(x.shape[1]),
            "n_active_features": int(grid["n_active_features"]),
            "n_anomalies": int(np.sum(y)),
            "anomaly_rate": float(np.mean(y)),
            "selection_metric": metric,
            "best_metric_value": float(row[metric]),
            "auc": float(row["auc"]),
            "ap": float(row["ap"]),
            "rho": float(row["rho"]),
            "requested_subspaces": int(row["requested_subspaces"]),
            "effective_subspaces": int(row["effective_subspaces"]),
            "requested_k": int(row["requested_k"]),
            "effective_k": int(row["effective_k"]),
            "selected_features": int(row["selected_features"]),
            "n_fine_balls": int(row["n_fine_balls"]),
            "n_coarse_balls": int(row["n_coarse_balls"]),
            "nested_entropy_max_error": float(
                grid["nested_entropy_max_error"]
            ),
            "model_seconds": float(grid["model_seconds"]),
            "metrics_seconds": float(metrics_seconds),
            "total_seconds": float(total_seconds),
            "input_sha256": input_hash,
            "result_npz": str(result_path),
        }

    return (
        make_best_summary(best_auc, "auc"),
        make_best_summary(best_ap, "ap"),
        detail_rows,
        diagnostics,
    )


def _parse_int_grid(value: str) -> List[int]:
    try:
        values = sorted(
            set(int(part.strip()) for part in value.split(",") if part.strip())
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid must contain integers") from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("integer grid values must be positive")
    return values


def _parse_rho_grid(value: str) -> List[float]:
    try:
        values = sorted(
            set(float(part.strip()) for part in value.split(",") if part.strip())
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rho grid must contain numbers") from exc
    if not values or any(item <= 0.0 or item > 1.0 for item in values):
        raise argparse.ArgumentTypeError("rho values must be in (0, 1]")
    return values


def _error_summary(dataset: str, metric: str, error: str) -> Dict[str, Any]:
    row = {field: None for field in BEST_FIELDS}
    row.update(
        {
            "dataset": dataset,
            "status": "error",
            "error": error,
            "selection_metric": metric,
        }
    )
    return row


def _write_checkpoint_outputs(
    output_dir: Path,
    metadata: Dict[str, Any],
    best_auc_rows: Sequence[Dict[str, Any]],
    best_ap_rows: Sequence[Dict[str, Any]],
    detail_rows: Sequence[Dict[str, Any]],
    completed: Sequence[str],
    status: str,
) -> None:
    _atomic_csv(
        output_dir / "grid_detail_auc_ap.csv",
        detail_rows,
        DETAIL_FIELDS,
    )
    _atomic_csv(
        output_dir / "best_auc_summary.csv",
        best_auc_rows,
        BEST_FIELDS,
    )
    _atomic_csv(
        output_dir / "best_ap_summary.csv",
        best_ap_rows,
        BEST_FIELDS,
    )
    # Stable aliases used by the independent result validator.
    _atomic_csv(
        output_dir / "best_by_auc.csv",
        best_auc_rows,
        BEST_FIELDS,
    )
    _atomic_csv(
        output_dir / "best_by_ap.csv",
        best_ap_rows,
        BEST_FIELDS,
    )
    ap_by_dataset = {row.get("dataset"): row for row in best_ap_rows}
    dataset_summary: List[Dict[str, Any]] = []
    for auc_row in best_auc_rows:
        dataset = auc_row.get("dataset")
        ap_row = ap_by_dataset.get(dataset, {})
        ok = auc_row.get("status") == "ok" and ap_row.get("status") == "ok"
        dataset_summary.append(
            {
                "dataset": dataset,
                "status": "ok" if ok else "error",
                "error": "" if ok else (auc_row.get("error") or ap_row.get("error")),
                "n_samples": auc_row.get("n_samples"),
                "n_original_features": auc_row.get("n_original_features"),
                "n_active_features": auc_row.get("n_active_features"),
                "n_anomalies": auc_row.get("n_anomalies"),
                "anomaly_rate": auc_row.get("anomaly_rate"),
                "best_auc": auc_row.get("auc"),
                "ap_at_best_auc": auc_row.get("ap"),
                "best_ap": ap_row.get("ap"),
                "auc_at_best_ap": ap_row.get("auc"),
                "nested_entropy_max_error": auc_row.get(
                    "nested_entropy_max_error"
                ),
                "input_sha256": auc_row.get("input_sha256"),
                "result_npz": auc_row.get("result_npz"),
            }
        )
    _atomic_csv(
        output_dir / "dataset_summary.csv",
        dataset_summary,
        DATASET_SUMMARY_FIELDS,
    )
    successful = [row for row in dataset_summary if row["status"] == "ok"]
    failed = [row for row in dataset_summary if row["status"] != "ok"]
    overview = {
        "method": metadata.get("method"),
        "status": status,
        "attempted_datasets": len(dataset_summary),
        "successful_datasets": len(successful),
        "failed_datasets": len(failed),
        "failed_dataset_names": [row["dataset"] for row in failed],
        "mean_best_auc": (
            float(np.mean([float(row["best_auc"]) for row in successful]))
            if successful
            else None
        ),
        "median_best_auc": (
            float(np.median([float(row["best_auc"]) for row in successful]))
            if successful
            else None
        ),
        "mean_best_ap": (
            float(np.mean([float(row["best_ap"]) for row in successful]))
            if successful
            else None
        ),
        "median_best_ap": (
            float(np.median([float(row["best_ap"]) for row in successful]))
            if successful
            else None
        ),
        "nested_entropy_max_error": (
            max(float(row["nested_entropy_max_error"]) for row in successful)
            if successful
            else None
        ),
        "selection_protocol": metadata.get("selection_protocol"),
        "score_direction": metadata.get("score_direction"),
        "elapsed_seconds": metadata.get("elapsed_seconds"),
    }
    _atomic_json(output_dir / "overview.json", overview)
    payload = {
        "status": status,
        "metadata": metadata,
        "completed_datasets": list(completed),
        "failed_datasets": [row["dataset"] for row in failed],
        "best_auc_summary": list(best_auc_rows),
        "best_ap_summary": list(best_ap_rows),
        "grid_detail": list(detail_rows),
    }
    _atomic_json(output_dir / "checkpoint.json", payload)
    _atomic_json(output_dir / "run_results.json", payload)


def _append_log(path: Path, message: str, also_print: bool = True) -> None:
    if also_print:
        print(message, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _snapshot_sources(output_dir: Path) -> None:
    snapshot = output_dir / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parent
    for source in sorted(source_root.glob("*.py")):
        shutil.copy2(str(source), str(snapshot / source.name))
    for name in ("README.md", "requirements.txt"):
        source = source_root / name
        if source.exists():
            shutil.copy2(str(source), str(snapshot / name))


def _run_contract(args: argparse.Namespace) -> Dict[str, Any]:
    """Return settings that must not change across a resumed experiment."""
    return {
        "data_dir": str(Path(args.data_dir).resolve()),
        "rho_grid": [float(value) for value in args.rho_grid],
        "subspace_grid": [int(value) for value in args.subspace_grid],
        "k_grid": [int(value) for value in args.k_grid],
        "seed": int(args.seed),
        "max_balls": args.max_balls,
        "relation_floor": float(args.relation_floor),
    }


def _validate_resume_contract(
    prior_metadata: Dict[str, Any], args: argparse.Namespace
) -> None:
    expected = _run_contract(args)
    differences: List[str] = []
    for key, value in expected.items():
        prior = prior_metadata.get(key, "<missing>")
        if key == "data_dir" and prior != "<missing>":
            prior = str(Path(str(prior)).resolve())
        if prior != value:
            differences.append("%s: saved=%r current=%r" % (key, prior, value))
    if differences:
        raise ValueError(
            "--resume settings differ from the checkpoint: "
            + "; ".join(differences)
        )


def _saved_dataset_artifacts_valid(
    output_dir: Path, dataset: str, expected_hash: str
) -> bool:
    """Cheap resume guard for the two atomically written dataset artifacts."""
    archive_path = output_dir / "per_dataset" / (dataset + ".npz")
    diagnostic_path = output_dir / "per_dataset" / (
        dataset + "_diagnostics.json"
    )
    if not archive_path.is_file() or not diagnostic_path.is_file():
        return False
    required = {
        "labels",
        "scores",
        "rho",
        "requested_subspaces",
        "requested_k",
        "auc",
        "ap",
        "input_sha256",
    }
    try:
        with np.load(str(archive_path), allow_pickle=False) as archive:
            if not required.issubset(set(archive.files)):
                return False
            hash_value = np.asarray(archive["input_sha256"])
            if hash_value.size != 1 or str(hash_value.reshape(-1)[0]) != expected_hash:
                return False
        with diagnostic_path.open("r", encoding="utf-8") as handle:
            diagnostics = json.load(handle)
        if diagnostics.get("dataset") != dataset:
            return False
        if diagnostics.get("input_sha256") != expected_hash:
            return False
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def run_batch(args: argparse.Namespace) -> int:
    files = sorted(Path(args.data_dir).glob("*.mat"))
    if not files:
        raise ValueError("no .mat files found under %s" % args.data_dir)
    if args.limit is None and args.expected_datasets is not None:
        if len(files) != int(args.expected_datasets):
            raise ValueError(
                "expected %d datasets but found %d"
                % (int(args.expected_datasets), len(files))
            )
    if args.limit is not None:
        files = files[: int(args.limit)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists() and not args.resume and not args.overwrite:
        raise ValueError(
            "checkpoint already exists; use --resume or a new output directory"
        )

    # Hash once up front.  The same manifest is used to validate resumed
    # per-dataset artifacts and is then written for the independent auditor.
    manifest_rows = [
        {
            "dataset": path.stem,
            "input_path": str(path.resolve()),
            "input_sha256": _sha256(path),
            "file_size_bytes": int(path.stat().st_size),
        }
        for path in files
    ]
    input_hashes = {
        str(row["dataset"]): str(row["input_sha256"]) for row in manifest_rows
    }

    best_auc_rows: List[Dict[str, Any]] = []
    best_ap_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    completed: List[str] = []
    if args.resume and checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        _validate_resume_contract(dict(checkpoint.get("metadata", {})), args)
        best_auc_rows = list(checkpoint.get("best_auc_summary", []))
        best_ap_rows = list(checkpoint.get("best_ap_summary", []))
        detail_rows = list(checkpoint.get("grid_detail", []))
        completed = list(checkpoint.get("completed_datasets", []))

    # Old checkpoints may have marked failed attempts as completed. Rebuild the
    # success set from status so --resume always retries failures lacking NPZs.
    successful_auc = {
        str(row.get("dataset"))
        for row in best_auc_rows
        if row.get("status") == "ok"
    }
    successful_ap = {
        str(row.get("dataset"))
        for row in best_ap_rows
        if row.get("status") == "ok"
    }
    successful_names = successful_auc.intersection(successful_ap)
    completed = [
        name
        for name in completed
        if name in successful_names
        and name in input_hashes
        and _saved_dataset_artifacts_valid(
            output_dir, name, input_hashes[name]
        )
    ]
    _atomic_csv(
        output_dir / "dataset_manifest.csv", manifest_rows, MANIFEST_FIELDS
    )

    metadata = {
        "method": "ZEKAD-Structural",
        "method_version": "1.0.0",
        **_run_contract(args),
        "selection_protocol": (
            "best AUC and best AP on the same labeled dataset are saved separately "
            "as oracle diagnostics; the detector itself is label free"
        ),
        "score_direction": "higher_is_anomaly",
        "label_informed_score_flip": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
        "command": sys.argv,
        "started_unix": time.time(),
    }
    environment = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
    }
    _atomic_json(output_dir / "run_config.json", metadata)
    _atomic_json(output_dir / "environment.json", environment)
    _snapshot_sources(output_dir)
    if not args.resume or not log_path.exists():
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("ZEKAD-Structural run\n")
    config = ModelConfig(
        seed=int(args.seed),
        max_balls=args.max_balls,
        relation_floor=float(args.relation_floor),
    )
    run_started = time.perf_counter()
    completed_set = set(completed)
    for position, path in enumerate(files, start=1):
        if path.stem in completed_set:
            _append_log(
                log_path,
                "[%d/%d] %s (already completed)"
                % (position, len(files), path.name),
            )
            continue
        # Replace an earlier failed attempt instead of duplicating it.
        best_auc_rows = [
            row for row in best_auc_rows if row.get("dataset") != path.stem
        ]
        best_ap_rows = [
            row for row in best_ap_rows if row.get("dataset") != path.stem
        ]
        detail_rows = [
            row for row in detail_rows if row.get("dataset") != path.stem
        ]
        _append_log(
            log_path,
            "[%d/%d] %s" % (position, len(files), path.name),
        )
        succeeded = False
        try:
            best_auc, best_ap, rows, diagnostics = evaluate_dataset_file(
                path=path,
                output_dir=output_dir,
                rho_grid=args.rho_grid,
                subspace_grid=args.subspace_grid,
                k_grid=args.k_grid,
                config=config,
            )
            best_auc_rows.append(best_auc)
            best_ap_rows.append(best_ap)
            detail_rows.extend(rows)
            succeeded = True
            _append_log(
                log_path,
                "  AUC=%.6f AP@AUC=%.6f | AP=%.6f AUC@AP=%.6f | nested_err=%.3g"
                % (
                    best_auc["auc"],
                    best_auc["ap"],
                    best_ap["ap"],
                    best_ap["auc"],
                    diagnostics["nested_entropy_max_error"],
                ),
            )
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
            best_auc_rows.append(_error_summary(path.stem, "auc", error))
            best_ap_rows.append(_error_summary(path.stem, "ap", error))
            _append_log(log_path, "  ERROR: %s" % error)
        if succeeded:
            completed.append(path.stem)
            completed_set.add(path.stem)
        metadata["elapsed_seconds"] = float(time.perf_counter() - run_started)
        metadata["successful_datasets"] = len(completed)
        metadata["attempted_datasets"] = len(best_auc_rows)
        _atomic_json(output_dir / "run_config.json", metadata)
        _write_checkpoint_outputs(
            output_dir,
            metadata,
            best_auc_rows,
            best_ap_rows,
            detail_rows,
            completed,
            "running",
        )
    metadata["elapsed_seconds"] = float(time.perf_counter() - run_started)
    metadata["finished_unix"] = time.time()
    failed_count = sum(row.get("status") != "ok" for row in best_auc_rows)
    final_status = "complete" if failed_count == 0 else "complete_with_errors"
    _atomic_json(output_dir / "run_config.json", metadata)
    _write_checkpoint_outputs(
        output_dir,
        metadata,
        best_auc_rows,
        best_ap_rows,
        detail_rows,
        completed,
        final_status,
    )
    _append_log(
        log_path,
        "Completed %d datasets in %.2fs" % (len(files), metadata["elapsed_seconds"]),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rho-grid",
        type=_parse_rho_grid,
        default=_parse_rho_grid("0.4,0.6,0.8,1.0"),
    )
    parser.add_argument(
        "--subspace-grid",
        type=_parse_int_grid,
        default=_parse_int_grid("5,10,20,30"),
    )
    parser.add_argument(
        "--k-grid",
        type=_parse_int_grid,
        default=_parse_int_grid("1,2,3,5,7,10,15,20,30,50"),
    )
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--max-balls", type=int, default=None)
    parser.add_argument("--relation-floor", type=float, default=RELATION_FLOOR)
    parser.add_argument("--expected-datasets", type=int, default=70)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_balls is not None and args.max_balls < 1:
        parser.error("--max-balls must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.expected_datasets is not None and args.expected_datasets < 1:
        parser.error("--expected-datasets must be positive")
    if args.relation_floor <= 0.0 or args.relation_floor > 1.0:
        parser.error("--relation-floor must be in (0, 1]")
    try:
        return run_batch(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
