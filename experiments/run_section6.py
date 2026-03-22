import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Tuple


STRATEGIES = [
    "random",
    "S_E",
    "S_NE",
    "S_W",
    "S_NW",
    "S_B",
    "S_NB",
    "S_EIG",
    "S_NEIG",
    "S_DU",
    "S_NDU",
    "S_DC",
    "S_NDC",
]


@dataclass
class ExperimentConfig:
    num_alternatives: int = 50
    num_criteria: int = 4
    num_feature_points: int = 4
    num_classes: int = 5
    num_dms: int = 3
    repetitions: int = 100
    stop_mode: str = "utility_stability"
    stop_threshold: float = 0.05
    stable_rounds: int = 3
    max_iterations: int = 15
    initial_queries: int = 5
    random_seed: int = 20260322
    threshold_jitter: float = 0.015
    dm_noise: float = 0.08
    utility_slack: float = 0.02
    eig_weight_entropy: float = 0.45
    eig_weight_width: float = 0.35
    eig_weight_concentration: float = 0.20
    neighbor_bandwidth: float = 0.75


@dataclass
class Alternative:
    idx: int
    criteria: List[float]
    true_utility: float
    true_class: int


@dataclass
class StateRecord:
    utility_low: float
    utility_high: float
    midpoint: float
    width: float
    entropy: float
    boundary: float
    confidence: float
    compatibility: List[float]


def normalize(values: Sequence[float]) -> List[float]:
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [v / total for v in values]


def piecewise_linear_value(x: float, knots: Sequence[float], values: Sequence[float]) -> float:
    if x <= knots[0]:
        return values[0]
    for left in range(len(knots) - 1):
        if x <= knots[left + 1]:
            x0, x1 = knots[left], knots[left + 1]
            y0, y1 = values[left], values[left + 1]
            if x1 == x0:
                return y1
            ratio = (x - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return values[-1]


def compute_class_thresholds(config: ExperimentConfig, rng: random.Random) -> List[float]:
    step = 1.0 / config.num_classes
    thresholds = [0.0]
    for h in range(1, config.num_classes):
        base = h * step
        jitter = rng.uniform(-config.threshold_jitter, config.threshold_jitter)
        thresholds.append(min(0.95, max(0.05, base + jitter)))
    thresholds.append(1.0)
    thresholds = sorted(thresholds)
    thresholds[0] = 0.0
    thresholds[-1] = 1.0
    return thresholds


def utility_to_class(utility: float, thresholds: Sequence[float]) -> int:
    for h in range(1, len(thresholds)):
        if utility < thresholds[h]:
            return h
    return len(thresholds) - 1


def class_midpoints(thresholds: Sequence[float]) -> List[float]:
    mids = []
    for h in range(1, len(thresholds)):
        mids.append((thresholds[h - 1] + thresholds[h]) / 2.0)
    return mids


def interval_support(lower: int, upper: int, num_classes: int) -> List[float]:
    support = [0.0] * num_classes
    width = upper - lower + 1
    for h in range(lower - 1, upper):
        support[h] = 1.0 / width
    return support


def aggregate_support(intervals: Sequence[Tuple[int, int]], num_classes: int) -> List[float]:
    accum = [0.0] * num_classes
    for lower, upper in intervals:
        contrib = interval_support(lower, upper, num_classes)
        for h, value in enumerate(contrib):
            accum[h] += value
    return normalize(accum)


def support_to_expected_utility(support: Sequence[float], thresholds: Sequence[float]) -> float:
    mids = class_midpoints(thresholds)
    return sum(p * mids[h] for h, p in enumerate(support))


def compatibility_from_interval(low: float, high: float, thresholds: Sequence[float]) -> List[float]:
    high = max(high, low + 1e-9)
    total = high - low
    values = []
    for h in range(1, len(thresholds)):
        overlap = max(0.0, min(high, thresholds[h]) - max(low, thresholds[h - 1]))
        values.append(overlap / total)
    return normalize(values)


def entropy(distribution: Sequence[float]) -> float:
    value = 0.0
    for p in distribution:
        if p > 1e-12:
            value -= p * math.log(p)
    return value


def l1_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(abs(a - b) for a, b in zip(left, right))


def euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def simulate_group_answer(alt: Alternative, thresholds: Sequence[float], config: ExperimentConfig, rng: random.Random) -> Tuple[List[Tuple[int, int]], List[float]]:
    intervals: List[Tuple[int, int]] = []
    for _ in range(config.num_dms):
        lower = alt.true_class
        upper = alt.true_class
        nearest_gap = min(
            abs(alt.true_utility - thresholds[h]) for h in range(1, len(thresholds) - 1)
        )
        boundary_pressure = max(0.0, 1.0 - nearest_gap / 0.15)
        noise = rng.random()
        if boundary_pressure > 0.45 or noise < config.dm_noise:
            if rng.random() < 0.5 and alt.true_class > 1:
                lower = alt.true_class - 1
            elif alt.true_class < config.num_classes:
                upper = alt.true_class + 1
        intervals.append((lower, upper))
    support = aggregate_support(intervals, config.num_classes)
    return intervals, support


def generate_dataset(config: ExperimentConfig, rng: random.Random) -> Tuple[List[Alternative], List[float]]:
    thresholds = compute_class_thresholds(config, rng)
    weights_raw = [rng.random() for _ in range(config.num_criteria)]
    weights = normalize(weights_raw)
    knots = [k / (config.num_feature_points - 1) for k in range(config.num_feature_points)]
    node_values: List[List[float]] = []
    for weight in weights:
        increments = [rng.random() for _ in range(config.num_feature_points - 1)]
        scale = weight / sum(increments)
        values = [0.0]
        running = 0.0
        for inc in increments:
            running += inc * scale
            values.append(running)
        values[-1] = weight
        node_values.append(values)

    alternatives: List[Alternative] = []
    for idx in range(config.num_alternatives):
        criteria = [rng.random() for _ in range(config.num_criteria)]
        utility = 0.0
        for j, x in enumerate(criteria):
            utility += piecewise_linear_value(x, knots, node_values[j])
        utility = max(0.0, min(1.0, utility))
        true_class = utility_to_class(utility, thresholds)
        alternatives.append(Alternative(idx=idx, criteria=criteria, true_utility=utility, true_class=true_class))
    return alternatives, thresholds


class FuzzyActiveLearner:
    def __init__(self, alternatives: Sequence[Alternative], thresholds: Sequence[float], config: ExperimentConfig, strategy: str, rng: random.Random):
        self.alternatives = list(alternatives)
        self.thresholds = list(thresholds)
        self.config = config
        self.strategy = strategy
        self.rng = rng
        self.queried: Dict[int, List[float]] = {}
        self.query_intervals: Dict[int, List[Tuple[int, int]]] = {}
        self.history: List[Dict[int, StateRecord]] = []
        self.consecutive_stable_rounds = 0
        self.distance_matrix = self._build_distance_matrix()

    def _build_distance_matrix(self) -> List[List[float]]:
        matrix: List[List[float]] = []
        for left in self.alternatives:
            row = []
            for right in self.alternatives:
                row.append(euclidean(left.criteria, right.criteria))
            matrix.append(row)
        return matrix

    def seed_initial_queries(self) -> None:
        by_class: Dict[int, List[Alternative]] = {}
        for alt in self.alternatives:
            by_class.setdefault(alt.true_class, []).append(alt)
        chosen: List[int] = []
        for class_id in range(1, self.config.num_classes + 1):
            pool = by_class.get(class_id, [])
            if pool:
                chosen.append(self.rng.choice(pool).idx)
        remaining = [alt.idx for alt in self.alternatives if alt.idx not in chosen]
        self.rng.shuffle(remaining)
        chosen.extend(remaining[: max(0, self.config.initial_queries - len(chosen))])
        for idx in chosen[: self.config.initial_queries]:
            self.query(idx)
        self.history.append(self.estimate_state())

    def query(self, idx: int) -> None:
        alt = self.alternatives[idx]
        intervals, support = simulate_group_answer(alt, self.thresholds, self.config, self.rng)
        self.queried[idx] = support
        self.query_intervals[idx] = intervals

    def estimate_state(self) -> Dict[int, StateRecord]:
        state: Dict[int, StateRecord] = {}
        labeled_ids = list(self.queried.keys())
        class_uniform = [1.0 / self.config.num_classes] * self.config.num_classes
        for alt in self.alternatives:
            if alt.idx in self.queried:
                support = list(self.queried[alt.idx])
                midpoint = support_to_expected_utility(support, self.thresholds)
                width = max(0.02, entropy(support) / max(1, self.config.num_classes) * 0.2)
                low = max(0.0, midpoint - width / 2.0)
                high = min(1.0, midpoint + width / 2.0)
                compatibility = compatibility_from_interval(low, high, self.thresholds)
            elif not labeled_ids:
                support = list(class_uniform)
                midpoint = 0.5
                width = 0.4
                low, high = 0.3, 0.7
                compatibility = list(class_uniform)
            else:
                weighted_support = [0.0] * self.config.num_classes
                weight_total = 0.0
                min_distance = None
                disagreement = 0.0
                for labeled_idx in labeled_ids:
                    labeled_alt = self.alternatives[labeled_idx]
                    dist = self.distance_matrix[alt.idx][labeled_idx]
                    min_distance = dist if min_distance is None else min(min_distance, dist)
                    weight = 1.0 / ((dist + 1e-6) ** 2)
                    weight_total += weight
                    labeled_support = self.queried[labeled_idx]
                    disagreement += entropy(labeled_support) * weight
                    for h, value in enumerate(labeled_support):
                        weighted_support[h] += weight * value
                support = normalize(weighted_support if weight_total > 0 else class_uniform)
                midpoint = support_to_expected_utility(support, self.thresholds)
                queried_ratio = len(labeled_ids) / len(self.alternatives)
                distance_factor = min(1.0, (min_distance or 0.0) / self.config.neighbor_bandwidth)
                disagreement_factor = min(1.0, disagreement / max(weight_total, 1e-6))
                uncertainty = 0.15 + 0.25 * distance_factor + 0.20 * disagreement_factor + 0.15 * (1.0 - queried_ratio)
                width = min(0.9, max(self.config.utility_slack, uncertainty))
                low = max(0.0, midpoint - width / 2.0)
                high = min(1.0, midpoint + width / 2.0)
                if high - low < self.config.utility_slack:
                    high = min(1.0, low + self.config.utility_slack)
                    low = max(0.0, high - self.config.utility_slack)
                compatibility = compatibility_from_interval(low, high, self.thresholds)
            ent = entropy(compatibility)
            width = high - low
            midpoint = (high + low) / 2.0
            boundary = 1.0 / (min(abs(midpoint - self.thresholds[h]) for h in range(1, len(self.thresholds) - 1)) + 1e-6)
            confidence = max(compatibility)
            state[alt.idx] = StateRecord(
                utility_low=low,
                utility_high=high,
                midpoint=midpoint,
                width=width,
                entropy=ent,
                boundary=boundary,
                confidence=confidence,
                compatibility=compatibility,
            )
        return state

    def current_uncertainty(self, state: Dict[int, StateRecord], unlabeled_ids: Sequence[int]) -> float:
        if not unlabeled_ids:
            return 0.0
        avg_entropy = sum(state[idx].entropy for idx in unlabeled_ids) / len(unlabeled_ids)
        avg_width = sum(state[idx].width for idx in unlabeled_ids) / len(unlabeled_ids)
        lack_concentration = sum(1.0 - state[idx].confidence for idx in unlabeled_ids) / len(unlabeled_ids)
        return (
            self.config.eig_weight_entropy * avg_entropy
            + self.config.eig_weight_width * avg_width
            + self.config.eig_weight_concentration * lack_concentration
        )

    def expected_information_gain(self, idx: int, state: Dict[int, StateRecord], unlabeled_ids: Sequence[int]) -> float:
        current = self.current_uncertainty(state, unlabeled_ids)
        if not unlabeled_ids:
            return 0.0
        record = state[idx]
        local_uncertainty = record.entropy + record.width + (1.0 - record.confidence)
        neighborhood = 0.0
        for other_idx in unlabeled_ids:
            dist = self.distance_matrix[idx][other_idx]
            neighborhood += math.exp(-dist / max(self.config.neighbor_bandwidth, 1e-6))
        neighborhood /= max(1, len(unlabeled_ids))
        expected_reduction = local_uncertainty * neighborhood * 0.25
        return max(0.0, min(current, expected_reduction))

    def choose_query(self, state: Dict[int, StateRecord]) -> int:
        unlabeled_ids = [alt.idx for alt in self.alternatives if alt.idx not in self.queried]
        if not unlabeled_ids:
            raise RuntimeError("No candidates left to query.")
        if self.strategy == "random":
            return self.rng.choice(unlabeled_ids)

        if len(self.history) >= 1:
            previous = self.history[-1]
        else:
            previous = state

        scores: Dict[int, float] = {}
        for idx in unlabeled_ids:
            record = state[idx]
            if self.strategy in {"S_E", "S_NE"}:
                scores[idx] = record.entropy
            elif self.strategy in {"S_W", "S_NW"}:
                scores[idx] = record.width
            elif self.strategy in {"S_B", "S_NB"}:
                scores[idx] = record.boundary
            elif self.strategy in {"S_EIG", "S_NEIG"}:
                scores[idx] = self.expected_information_gain(idx, state, unlabeled_ids)
            elif self.strategy in {"S_DU", "S_NDU"}:
                prev = previous[idx]
                scores[idx] = abs(record.utility_low - prev.utility_low) + abs(record.utility_high - prev.utility_high)
            elif self.strategy in {"S_DC", "S_NDC"}:
                prev = previous[idx]
                scores[idx] = l1_distance(record.compatibility, prev.compatibility)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

        reverse = self.strategy in {"S_E", "S_W", "S_B", "S_EIG", "S_DU", "S_DC"}
        ordered = sorted(scores.items(), key=lambda item: (item[1], -item[0]), reverse=reverse)
        return ordered[0][0]

    def check_stop(self, state: Dict[int, StateRecord]) -> bool:
        if len(self.history) < 1:
            return False
        previous = self.history[-1]
        deltas = [
            abs(state[idx].utility_low - previous[idx].utility_low) + abs(state[idx].utility_high - previous[idx].utility_high)
            for idx in state
        ]
        delta_u = sum(deltas) / len(deltas)
        if self.config.stop_mode == "utility_stability":
            if delta_u <= self.config.stop_threshold:
                self.consecutive_stable_rounds += 1
            else:
                self.consecutive_stable_rounds = 0
            return self.consecutive_stable_rounds >= self.config.stable_rounds
        raise ValueError(f"Unsupported stop mode: {self.config.stop_mode}")

    def evaluate(self, state: Dict[int, StateRecord]) -> Dict[str, float]:
        accuracy = 0
        adjacent_accuracy = 0
        avg_entropy = 0.0
        avg_width = 0.0
        support_rmse = 0.0
        for alt in self.alternatives:
            predicted_class = state[alt.idx].compatibility.index(max(state[alt.idx].compatibility)) + 1
            if predicted_class == alt.true_class:
                accuracy += 1
            if abs(predicted_class - alt.true_class) <= 1:
                adjacent_accuracy += 1
            avg_entropy += state[alt.idx].entropy
            avg_width += state[alt.idx].width
            true_support = interval_support(alt.true_class, alt.true_class, self.config.num_classes)
            diff = sum((state[alt.idx].compatibility[h] - true_support[h]) ** 2 for h in range(self.config.num_classes))
            support_rmse += math.sqrt(diff / self.config.num_classes)
        total = len(self.alternatives)
        queries_used = float(len(self.queried))
        accuracy_value = accuracy / total
        return {
            "accuracy": accuracy_value,
            "queries_used": queries_used,
            "accuracy_per_question": accuracy_value / max(queries_used, 1.0),
        }

    def run(self) -> Dict[str, float]:
        self.seed_initial_queries()
        last_state = self.history[-1]
        while len(self.queried) < len(self.alternatives) and len(self.history) - 1 < self.config.max_iterations:
            chosen_idx = self.choose_query(last_state)
            self.query(chosen_idx)
            state = self.estimate_state()
            should_stop = self.check_stop(state)
            self.history.append(state)
            last_state = state
            if should_stop:
                break
        metrics = self.evaluate(last_state)
        metrics["iterations"] = float(len(self.history) - 1)
        return metrics


def run_batch(config: ExperimentConfig, output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    rows: List[Dict[str, float]] = []
    started_at = time.time()
    for repetition in range(config.repetitions):
        dataset_rng = random.Random(config.random_seed + repetition)
        alternatives, thresholds = generate_dataset(config, dataset_rng)
        for strategy_idx, strategy in enumerate(STRATEGIES):
            run_rng = random.Random(config.random_seed + repetition * 1000 + strategy_idx)
            learner = FuzzyActiveLearner(alternatives, thresholds, config, strategy, run_rng)
            metrics = learner.run()
            row: Dict[str, float] = {
                "repetition": float(repetition),
                "strategy_index": float(strategy_idx),
                "strategy": strategy,
            }
            row.update(metrics)
            rows.append(row)

    csv_path = os.path.join(output_dir, "section6_strategy_comparison_runs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "repetition",
                "strategy_index",
                "strategy",
                "accuracy",
                "queries_used",
                "iterations",
                "accuracy_per_question",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    grouped: Dict[str, Dict[str, float]] = {}
    metric_fields = ["accuracy", "queries_used", "iterations", "accuracy_per_question"]
    for strategy in STRATEGIES:
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        grouped[strategy] = {}
        for field in metric_fields:
            grouped[strategy][field] = sum(float(row[field]) for row in strategy_rows) / len(strategy_rows)

    summary_path = os.path.join(output_dir, "section6_strategy_comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(
            {
                "config": asdict(config),
                "strategies": STRATEGIES,
                "num_runs": len(rows),
                "elapsed_seconds": time.time() - started_at,
                "summary": grouped,
            },
            summary_file,
            indent=2,
        )

    return csv_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the section 6 strategy-comparison experiment.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for result files.")
    parser.add_argument("--repetitions", type=int, default=100, help="Number of repetitions per strategy.")
    parser.add_argument("--seed", type=int, default=20260322, help="Global random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(repetitions=args.repetitions, random_seed=args.seed)
    csv_path, summary_path = run_batch(config, args.output_dir)
    print(f"Saved run-level results to {csv_path}")
    print(f"Saved summary results to {summary_path}")


if __name__ == "__main__":
    main()
