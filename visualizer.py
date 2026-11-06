"""Визуализация результатов трекинга: сопоставления, ключевые кадры и сводные графики"""

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

_MAX_MATCHES_TO_DRAW = 80
_KEYFRAMES_IN_GRID = 6


def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Переводит кадр из BGR в RGB для отображения в Matplotlib"""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _prepare_mask(inlier_mask, matches_count: int) -> list[int] | None:
    """Приводит маску инлайеров к длине отображаемых совпадений"""
    if inlier_mask is None or len(inlier_mask) == 0:
        return None

    mask = [int(flag) for flag in inlier_mask[:matches_count]]
    mask.extend([0] * (matches_count - len(mask)))
    return mask


def save_matches_visualization(
    ref_frame: np.ndarray,
    ref_kp: list[cv2.KeyPoint],
    target_frame: np.ndarray,
    target_kp: list[cv2.KeyPoint],
    good_matches: list[cv2.DMatch],
    inlier_mask: list[int] | None,
    corners: np.ndarray | None,
    output_path: Path,
    title: str = "",
) -> None:
    """Сохраняет сопоставления ключевых точек с рамкой найденного объекта"""
    target_copy = target_frame.copy()
    if corners is not None:
        cv2.polylines(target_copy, [np.int32(corners).reshape(-1, 1, 2)], True, (0, 255, 0), 3)

    shown_matches = list(good_matches)[:_MAX_MATCHES_TO_DRAW]
    matches_image = cv2.drawMatches(
        ref_frame,
        ref_kp,
        target_copy,
        target_kp,
        shown_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 100, 0),
        matchesMask=_prepare_mask(inlier_mask, len(shown_matches)),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    figure, axes = plt.subplots(figsize=(16, 9))
    axes.imshow(_to_rgb(matches_image))
    axes.set_title(title or "Сопоставление ключевых точек")
    axes.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_video_keyframes_grid(
    keyframes: list[tuple[int, np.ndarray]],
    output_path: Path,
    video_title: str,
) -> None:
    """Сохраняет сетку ключевых кадров видео (до шести штук)"""
    keyframes = list(keyframes)[:_KEYFRAMES_IN_GRID]
    if not keyframes:
        return

    count = len(keyframes)
    rows, cols = (2, 3) if count == _KEYFRAMES_IN_GRID else (1, count)

    figure, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, (frame_idx, frame_bgr) in zip(axes, keyframes):
        axis.imshow(_to_rgb(frame_bgr))
        axis.set_title(f"Кадр #{frame_idx}")
        axis.axis("off")

    for axis in axes[count:]:
        axis.axis("off")

    figure.suptitle(video_title)
    figure.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3.0)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_tracking_dynamics(video_stats: list[dict], output_path: Path) -> None:
    """Рисует покадровую динамику совпадений и инлайеров для каждого видео"""
    if not video_stats:
        return

    figure, axes = plt.subplots(
        len(video_stats),
        1,
        figsize=(11, 3.2 * len(video_stats)),
        squeeze=False,
    )

    for axis, stats in zip(axes.ravel(), video_stats):
        frames = stats.get("frame_indices", [])
        axis.plot(frames, stats.get("frame_good_matches", []), color="#1f77b4", label="Хорошие совпадения")
        axis.plot(frames, stats.get("frame_inliers", []), color="#d62728", label="Инлайеры RANSAC")
        axis.set_title(f"{stats.get('video_name', '')} ({stats.get('method', '').upper()})")
        axis.set_xlabel("Номер кадра")
        axis.set_ylabel("Количество точек")
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_detector_comparison(comparison_stats: list[dict], output_path: Path) -> None:
    """Сравнивает SIFT и ORB по доле обнаружений, инлайерам и скорости"""
    if not comparison_stats:
        return

    methods = [
        (name, name.upper(), color)
        for name, color in (("sift", "#1f77b4"), ("orb", "#ff7f0e"))
        if any(name in item for item in comparison_stats)
    ]
    if not methods:
        return

    metrics = [
        ("detection_rate_pct", "Доля кадров с объектом, %"),
        ("mean_inliers", "Среднее число инлайеров"),
        ("fps", "Скорость обработки, кадров/с"),
    ]
    video_names = [item.get("video_name", "") for item in comparison_stats]
    positions = np.arange(len(video_names))
    width = 0.8 / len(methods)

    figure, axes = plt.subplots(1, len(metrics), figsize=(5.6 * len(metrics), 4.8))
    for axis, (key, title) in zip(np.atleast_1d(axes), metrics):
        for index, (name, label, color) in enumerate(methods):
            values = [item.get(name, {}).get(key, 0.0) for item in comparison_stats]
            offset = (index - (len(methods) - 1) / 2) * width
            axis.bar(positions + offset, values, width, label=label, color=color)
        axis.set_title(title)
        axis.set_xticks(positions)
        axis.set_xticklabels(video_names, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.3)
        axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_summary_charts(
    video_stats: list[dict],
    comparison_stats: list[dict],
    output_dir: Path,
) -> None:
    """Сохраняет графики динамики трекинга и сравнения детекторов"""
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_tracking_dynamics(video_stats, output_dir / "tracking_dynamics.png")
    _plot_detector_comparison(comparison_stats, output_dir / "detector_comparison.png")
