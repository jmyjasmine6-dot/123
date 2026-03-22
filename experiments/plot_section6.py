import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Sequence

METRICS = [
    ("accuracy", "Accuracy"),
    ("iterations", "Iterations"),
    ("accuracy_per_question", "Accuracy / number of questions"),
]


def read_runs(csv_path: str) -> Dict[str, Dict[str, List[float]]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            strategy = row["strategy"]
            for metric, _ in METRICS:
                grouped[strategy][metric].append(float(row[metric]))
    return grouped


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def metric_stats(values: Sequence[float]) -> Dict[str, float]:
    mu = mean(values)
    se = std(values) / math.sqrt(len(values)) if values else 0.0
    ci = 1.96 * se
    return {
        "mean": mu,
        "se_low": mu - se,
        "se_high": mu + se,
        "ci_low": mu - ci,
        "ci_high": mu + ci,
    }


def svg_boxplot(grouped: Dict[str, Dict[str, List[float]]], metric: str, title: str, out_path: str) -> None:
    strategies = list(grouped.keys())
    stats = {strategy: metric_stats(grouped[strategy][metric]) for strategy in strategies}
    y_min = min(value["ci_low"] for value in stats.values())
    y_max = max(value["ci_high"] for value in stats.values())
    if abs(y_max - y_min) < 1e-9:
        y_min -= 1.0
        y_max += 1.0
    padding = 0.08 * (y_max - y_min)
    y_min -= padding
    y_max += padding

    width, height = 1280, 720
    left, right, top, bottom = 100, 40, 80, 140
    plot_width = width - left - right
    plot_height = height - top - bottom

    def y_to_px(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    x_step = plot_width / max(1, len(strategies))
    box_width = min(46, x_step * 0.45)
    cap_width = box_width * 0.75

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="40" text-anchor="middle" font-size="26" font-family="Arial">{title}</text>')
    parts.append(f'<text x="{width/2}" y="68" text-anchor="middle" font-size="14" fill="#555" font-family="Arial">Red diamond = mean, box = SE, whiskers = 95% CI</text>')

    # axes
    x0 = left
    y0 = top + plot_height
    parts.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{y0}" stroke="#222" stroke-width="2"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{left + plot_width}" y2="{y0}" stroke="#222" stroke-width="2"/>')

    # y ticks
    for tick_idx in range(6):
        value = y_min + tick_idx * (y_max - y_min) / 5.0
        y = y_to_px(value)
        parts.append(f'<line x1="{x0-8}" y1="{y}" x2="{left + plot_width}" y2="{y}" stroke="#ddd" stroke-width="1"/>')
        parts.append(f'<text x="{x0-14}" y="{y+5}" text-anchor="end" font-size="14" font-family="Arial">{value:.3f}</text>')

    for idx, strategy in enumerate(strategies):
        center_x = left + x_step * (idx + 0.5)
        stat = stats[strategy]
        y_ci_low = y_to_px(stat["ci_low"])
        y_ci_high = y_to_px(stat["ci_high"])
        y_se_low = y_to_px(stat["se_low"])
        y_se_high = y_to_px(stat["se_high"])
        y_mean = y_to_px(stat["mean"])

        # whisker + caps
        parts.append(f'<line x1="{center_x}" y1="{y_ci_low}" x2="{center_x}" y2="{y_ci_high}" stroke="#333" stroke-width="2"/>')
        parts.append(f'<line x1="{center_x-cap_width/2}" y1="{y_ci_low}" x2="{center_x+cap_width/2}" y2="{y_ci_low}" stroke="#333" stroke-width="2"/>')
        parts.append(f'<line x1="{center_x-cap_width/2}" y1="{y_ci_high}" x2="{center_x+cap_width/2}" y2="{y_ci_high}" stroke="#333" stroke-width="2"/>')

        # SE box
        rect_y = min(y_se_low, y_se_high)
        rect_h = max(1.0, abs(y_se_high - y_se_low))
        parts.append(f'<rect x="{center_x-box_width/2}" y="{rect_y}" width="{box_width}" height="{rect_h}" fill="#9ecae1" stroke="#3182bd" stroke-width="2"/>')

        # mean diamond
        diamond_w = box_width * 0.7
        diamond_h = 16
        points = [
            (center_x, y_mean - diamond_h / 2),
            (center_x + diamond_w / 2, y_mean),
            (center_x, y_mean + diamond_h / 2),
            (center_x - diamond_w / 2, y_mean),
        ]
        points_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(f'<polygon points="{points_text}" fill="#d62728" stroke="#8c1515" stroke-width="1.5"/>')

        parts.append(f'<text x="{center_x}" y="{y0+28}" text-anchor="middle" font-size="13" font-family="Arial" transform="rotate(35 {center_x} {y0+28})">{strategy}</text>')

    parts.append(f'<text x="24" y="{top + plot_height/2}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 24 {top + plot_height/2})">{title}</text>')
    parts.append('</svg>')

    with open(out_path, 'w', encoding='utf-8') as handle:
        handle.write("\n".join(parts))


def main() -> None:
    csv_path = os.path.join('outputs', 'section6_run', 'section6_strategy_comparison_runs.csv')
    plot_dir = os.path.join('outputs', 'section6_run', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    grouped = read_runs(csv_path)
    for metric, title in METRICS:
        svg_boxplot(grouped, metric, title, os.path.join(plot_dir, f'{metric}_boxplot.svg'))
    print(f'Created plots in {plot_dir}')


if __name__ == '__main__':
    main()
