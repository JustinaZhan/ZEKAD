"""ZEKAD: nested Zentropy granular-ball scores with KNN/LOF rank fusion.

All scoring functions accept features only. Larger scores indicate anomalies.
The 160 structural configurations retain equal weight, including effective
configurations that coincide on low-dimensional data.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.stats import rankdata

EPS = 1.0e-12
RELATION_FLOOR = 1.0e-15
IDENTITY_TOL = 1.0e-12
RHO_GRID = (0.4, 0.6, 0.8, 1.0)
SUBSPACE_GRID = (5, 10, 20, 30)
BALL_K_GRID = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50)
DENSITY_K_GRID = tuple(range(2, 61))
BETA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ALPHA = 0.20
DENSITY_STD_THRESHOLD = 1.0e-8

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


def structural_scores(
    x_raw: np.ndarray,
    dataset_name: str = "dataset",
    config: Optional[ModelConfig] = None,
) -> np.ndarray:
    """Return the paper's 160 structural score vectors (configurations × samples).

    Reuse each feature-prefix hierarchy and subspace bank, while retaining all
    requested rho/k/M combinations in their original order and multiplicity.
    """
    if config is None:
        config = ModelConfig()
    x, _ = preprocess_features(x_raw)
    active_features = x.shape[1]
    reference = build_hierarchical_balls(x, config.max_balls)
    reference_total = _pairwise_total_squared(reference.fine_centers)
    reference_sigma = _bandwidth_squared(reference_total, active_features)
    relevance = np.asarray([
        nested_zentropy_three_state(
            centers=reference.fine_centers,
            sizes=reference.fine_sizes,
            fine_to_coarse=reference.fine_to_coarse,
            b_features=[feature],
            sigma_squared=reference_sigma,
            relation_floor=config.relation_floor,
            total_squared=reference_total,
        ).relevance
        for feature in range(active_features)
    ])
    codes, entropies, contrasts = _discretize_features(x)
    nmi = _normalized_mi_matrix(codes, entropies)
    tolerance = 64.0 * np.finfo(np.float64).eps
    if float(np.sum(relevance)) > tolerance:
        selection_relevance = relevance.copy()
    elif float(np.sum(contrasts)) > tolerance:
        selection_relevance = contrasts.copy()
    else:
        selection_relevance = np.ones(active_features, dtype=np.float64)
    feature_order, _ = _strict_feature_path(selection_relevance, nmi)
    retain_counts = [
        min(active_features, max(2 if active_features > 1 else 1,
                                 int(math.ceil(rho * active_features))))
        for rho in RHO_GRID
    ]
    prefixes = {}
    for count in sorted(set(retain_counts)):
        selected = x[:, feature_order[:count].copy()]
        hierarchy = build_hierarchical_balls(selected, config.max_balls)
        rng = np.random.default_rng(
            _stable_seed(config.seed, dataset_name, "subspaces", count)
        )
        subspaces = _generate_subspaces(count, max(SUBSPACE_GRID), rng)
        caches, _ = _prepare_subspace_caches(
            selected, hierarchy, subspaces, config.relation_floor
        )
        prefixes[count] = hierarchy, caches

    rows = []
    for count in retain_counts:
        hierarchy, caches = prefixes[count]
        for k in BALL_K_GRID:
            components, _ = _subspace_scores_for_k(
                caches, hierarchy.fine_sizes.astype(np.float64),
                hierarchy.fine_membership, k,
            )
            for m in SUBSPACE_GRID:
                score, _, _, _ = _fuse_subspace_prefix(components, caches, m)
                rows.append(score)
    return np.vstack(rows)


def midrank01(values: np.ndarray) -> np.ndarray:
    """Average tied ranks: (rank - 0.5) / n; constant vectors map to 0.5."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError("cannot rank an empty vector")
    _require_finite(vector, "rank values")
    return (rankdata(vector, method="average") - 0.5) / float(vector.size)


def _density_scores(x_raw: np.ndarray, k_values: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Native PyOD training scores on unscaled features with std > 1e-8."""
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF

    data = np.asarray(x_raw, dtype=np.float64)
    kept = np.std(data, axis=0) > DENSITY_STD_THRESHOLD
    if not np.any(kept):
        raise ValueError("no features remain after the std>1e-8 density filter")
    data = np.ascontiguousarray(data[:, kept])
    common = dict(
        contamination=0.1, algorithm="auto", leaf_size=30,
        metric="minkowski", p=2, n_jobs=1,
    )
    knn_rows, lof_rows = [], []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        for k in k_values:
            random.seed(42)
            np.random.seed(42)
            knn = KNN(n_neighbors=k, method="largest", **common).fit(data)
            knn_score = np.asarray(knn.decision_scores_, dtype=np.float64)
            random.seed(42)
            np.random.seed(42)
            lof = LOF(n_neighbors=k, **common).fit(data)
            lof_score = np.asarray(lof.decision_scores_, dtype=np.float64)
            knn_rows.append(midrank01(knn_score))
            lof_rows.append(midrank01(lof_score))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return np.vstack(knn_rows), np.vstack(lof_rows)


def score_grid(
    x: np.ndarray,
    dataset_name: str = "dataset",
    k_grid: Sequence[int] = DENSITY_K_GRID,
    beta_grid: Sequence[float] = BETA_GRID,
) -> Dict[str, np.ndarray]:
    """Return label-free scores for the paper's density/fusion grid.

    ``scores`` has shape (configurations, samples), ordered by increasing k,
    then beta. ``config_k`` and ``config_beta`` identify each row. Dataset names
    enter the structural random seed; use the original MAT filename stem for
    reproduction. The full default grid requires at least 61 samples.
    """
    data = np.asarray(x)
    if data.ndim != 2 or data.shape[0] < 3 or data.shape[1] < 1:
        raise ValueError("X must contain at least three samples and one feature")
    if np.iscomplexobj(data) or not np.issubdtype(data.dtype, np.number):
        raise ValueError("X must be a real numeric matrix")
    data = np.asarray(data, dtype=np.float64)
    _require_finite(data, "features")
    requested_k = list(k_grid)
    if not requested_k or any(
        not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_))
        or k < 2 or k >= len(data) for k in requested_k
    ):
        raise ValueError("k values must be integers with 2 <= k < n_samples")
    k_values = sorted(set(int(k) for k in requested_k))
    beta_values = sorted(set(float(beta) for beta in beta_grid))
    if not beta_values or any(not np.isfinite(b) or b < 0 or b > 1 for b in beta_values):
        raise ValueError("beta values must be finite and in [0, 1]")
    # The 160 requested structural rows stay equally weighted, even when some
    # effective configurations coincide. Deduplication changes the estimator.
    structural = structural_scores(data, dataset_name=dataset_name)
    consensus = np.mean(np.vstack([midrank01(row) for row in structural]), axis=0)
    knn, lof = _density_scores(data, k_values)
    rows, config_k, config_beta = [], [], []
    for index, k in enumerate(k_values):
        for beta in beta_values:
            density = (1.0 - beta) * knn[index] + beta * lof[index]
            score = (1.0 - ALPHA) * density + ALPHA * consensus
            _require_finite(score, "ZEKAD score")
            rows.append(score)
            config_k.append(k)
            config_beta.append(beta)
    return {
        "scores": np.vstack(rows),
        "config_k": np.asarray(config_k, dtype=np.int64),
        "config_beta": np.asarray(config_beta, dtype=np.float64),
        "z_consensus": consensus,
    }


def score_samples(
    x: np.ndarray, dataset_name: str = "dataset", k: int = 5, beta: float = 0.0,
) -> np.ndarray:
    """Score the input samples at a fixed (k, beta), without using labels.

    This is transductive: each call constructs the hierarchy and fits the
    density models on the supplied X. (5, 0) is the default fixed-parameter
    example. The paper evaluates all 24 datasets with score_grid and selects
    AUC/AP maxima separately after score generation.
    """
    return score_grid(x, dataset_name, k_grid=(k,), beta_grid=(beta,))["scores"][0]
