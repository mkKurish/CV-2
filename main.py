"""Запуск трекинга плоского объекта на видео, сбор метрик и визуализация"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from tracker import PlanarObjectTracker
from visualizer import (
    save_matches_visualization,
    save_summary_charts,
    save_video_keyframes_grid,
)

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_KEYFRAME_COUNT = 6
_FALLBACK_KEYFRAME_STEP = 60
_SERIES_KEYS = {"frame_indices", "frame_good_matches", "frame_inliers", "frame_detected"}


def _mean(values: list[float]) -> float:
    """Среднее значение списка или ноль для пустого списка"""
    return float(np.mean(values)) if len(values) else 0.0


def _pick_evenly(items: list, count: int) -> list:
    """Берёт не более count элементов, равномерно распределённых по списку"""
    if len(items) <= count:
        return items
    step = len(items) / count
    return [items[int(index * step)] for index in range(count)]


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter | None:
    """Создаёт VideoWriter с кодеком mp4v или возвращает None при ошибке"""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        print(f"Предупреждение: не удалось открыть VideoWriter для {path}")
        return None
    return writer


def process_single_video(
    video_path: Path,
    output_dir: Path,
    method: str = "sift",
    label: str | None = None,
    save_artifacts: bool = True,
) -> dict:
    """Прогоняет трекер по видео и возвращает сводные метрики и массивы по кадрам"""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise ValueError(f"Не удалось прочитать первый кадр: {video_path}")

    # Реальный размер кадра важнее свойства контейнера — по нему создаётся VideoWriter
    height, width = first_frame.shape[:2]

    tracker = PlanarObjectTracker(
        first_frame,
        label=label or f"Object ({video_path.stem})",
        method=method,
    )

    output_video_path = output_dir / "videos" / f"{video_path.stem}_tracked.mp4"
    writer = _open_writer(output_video_path, fps, (width, height)) if save_artifacts else None

    # Равномерно распределённые по видео индексы кадров
    keyframe_indices = set([] if total_frames <= 0 else sorted({int(round(value)) for value in np.linspace(0, total_frames - 1, _KEYFRAME_COUNT)}))
    keyframes: list[tuple[int, np.ndarray]] = []
    sparse_keyframes: list[tuple[int, np.ndarray]] = []

    frame_indices: list[int] = []
    frame_good_matches: list[int] = []
    frame_inliers: list[int] = []
    frame_detected: list[bool] = []
    keypoints_counts: list[int] = []
    inlier_ratios: list[float] = []
    processing_times: list[float] = []

    best_score = (-1, -1, -1, -1)
    best_idx = -1
    best_frame = None
    best_result = None

    frame = first_frame
    frame_idx = 0
    while frame is not None:
        annotated, result = tracker.track_frame(frame, frame_idx)
        if writer is not None:
            writer.write(annotated)

        frame_indices.append(frame_idx)
        frame_good_matches.append(int(result.good_matches_count))
        frame_inliers.append(int(result.inliers_count))
        frame_detected.append(bool(result.detected))
        keypoints_counts.append(int(result.keypoints_count))
        inlier_ratios.append(float(result.inlier_ratio))
        processing_times.append(float(result.processing_time_ms))

        # Кадр для сетки ключевых кадров: заранее рассчитанный или резервный шаг
        if frame_idx in keyframe_indices:
            keyframes.append((frame_idx, annotated.copy()))
        elif not keyframe_indices and frame_idx % _FALLBACK_KEYFRAME_STEP == 0:
            sparse_keyframes.append((frame_idx, annotated.copy()))

        # Первый кадр — эталон, поэтому лучший кадр ищем только после него;
        # приоритет у кадров с прямым SIFT/ORB-обнаружением в сцене на столе
        if frame_idx > 0:
            score = (
                int(result.detected),
                int(result.homography is not None),
                int(frame_idx >= 50),
                int(result.inliers_count),
                int(result.good_matches_count),
            )
            if score > best_score:
                best_score = score
                best_idx = frame_idx
                best_frame = frame.copy()
                best_result = result

        frame_idx += 1
        ok, frame = capture.read()
        if not ok:
            frame = None

    capture.release()
    if writer is not None:
        writer.release()

    if not keyframes:
        keyframes = _pick_evenly(sparse_keyframes, _KEYFRAME_COUNT)

    keyframes_image = None
    matches_image = None
    if save_artifacts:
        if keyframes:
            keyframes_image = output_dir / f"keyframes_{video_path.stem}.png"
            save_video_keyframes_grid(
                keyframes,
                keyframes_image,
                f"Ключевые кадры: {video_path.stem} ({method.upper()})",
            )

        if best_frame is not None and best_result is not None and best_result.detected:
            matches_image = output_dir / f"matches_{video_path.stem}.png"
            # Совпадения пересчитываются заново: индексы queryIdx/trainIdx должны
            # соответствовать именно тем спискам точек, что уйдут в визуализатор
            target_kp, target_desc = tracker.detector.detect_and_compute(best_frame)
            good_matches = tracker.detector.match_descriptors(
                tracker.ref_descriptors, target_desc
            )
            inlier_mask = None
            if len(good_matches) >= 4:
                src_pts = np.float32(
                    [tracker.ref_keypoints[match.queryIdx].pt for match in good_matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [target_kp[match.trainIdx].pt for match in good_matches]
                ).reshape(-1, 1, 2)
                _, mask = cv2.findHomography(
                    src_pts,
                    dst_pts,
                    cv2.RANSAC,
                    tracker.ransac_thresh,
                    maxIters=3000,
                    confidence=0.995,
                )
                if mask is not None:
                    inlier_mask = mask.ravel().astype(int).tolist()

            save_matches_visualization(
                tracker.reference_frame,
                tracker.ref_keypoints,
                best_frame,
                target_kp,
                good_matches,
                inlier_mask,
                best_result.corners,
                matches_image,
                title=f"Лучшее сопоставление — кадр #{best_idx} ({video_path.stem})",
            )

    tracked_frames = len(frame_detected)
    mean_processing_ms = _mean(processing_times)
    detected_frames = sum(frame_detected)

    return {
        "video_name": video_path.name,
        "method": method.lower(),
        "resolution": f"{width}x{height}",
        "video_fps": round(fps, 2),
        "total_frames": total_frames if total_frames > 0 else tracked_frames,
        "tracked_frames": tracked_frames,
        "detection_rate_pct": round(100.0 * detected_frames / tracked_frames, 2) if tracked_frames else 0.0,
        "ref_keypoints": getattr(tracker, "base_ref_keypoints_count", len(tracker.ref_keypoints)),
        "mean_keypoints": round(_mean(keypoints_counts), 2),
        "mean_good_matches": round(_mean(frame_good_matches), 2),
        "mean_inliers": round(_mean(frame_inliers), 2),
        "max_inliers": max(frame_inliers, default=0),
        "mean_inlier_ratio": round(_mean(inlier_ratios), 4),
        "mean_processing_ms": round(mean_processing_ms, 2),
        "processing_fps": round(1000.0 / mean_processing_ms, 2) if mean_processing_ms > 0 else 0.0,
        "output_video": str(output_video_path) if writer is not None else None,
        "keyframes_image": str(keyframes_image) if keyframes_image else None,
        "matches_image": str(matches_image) if matches_image else None,
        "frame_indices": frame_indices,
        "frame_good_matches": frame_good_matches,
        "frame_inliers": frame_inliers,
        "frame_detected": frame_detected,
    }


def _find_videos(videos_dir: Path) -> list[Path]:
    """Возвращает видеофайлы каталога, отсортированные по имени"""
    if not videos_dir.is_dir():
        return []
    return sorted(
        (path for path in videos_dir.iterdir() if path.suffix.lower() in _VIDEO_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


def _build_comparison(video_name: str, primary: dict, secondary: dict) -> dict:
    """Готовит метрики двух прогонов для сравнительной диаграммы"""
    entry = {"video_name": video_name}
    for stats in (primary, secondary):
        entry[stats["method"]] = {
            "detection_rate_pct": stats["detection_rate_pct"],
            "mean_inliers": stats["mean_inliers"],
            "fps": stats["processing_fps"],
        }
    return entry


def _save_metrics(primary_stats: list[dict], comparison_stats: list[dict], output_dir: Path) -> None:
    """Сохраняет сводные метрики в компактный JSON без покадровых списков"""
    summary = {
        "videos": [
            {key: value for key, value in stats.items() if key not in _SERIES_KEYS}
            for stats in primary_stats
        ],
        "comparison": comparison_stats,
    }
    path = output_dir / "metrics.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Метрики сохранены: {path}")


def _print_table(primary_stats: list[dict], comparison_stats: list[dict]) -> None:
    """Выводит сводную таблицу результатов в консоль"""
    header = (
        f"{'Видео':<30}{'Метод':<7}{'Кадры':>7}{'Детект.,%':>11}"
        f"{'Совпад.':>10}{'Инлайеры':>10}{'FPS':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for stats in primary_stats:
        print(
            f"{stats['video_name'][:29]:<30}{stats['method'].upper():<7}"
            f"{stats['tracked_frames']:>7}{stats['detection_rate_pct']:>11.1f}"
            f"{stats['mean_good_matches']:>10.1f}{stats['mean_inliers']:>10.1f}"
            f"{stats['processing_fps']:>8.1f}"
        )

    comparison_header = f"{'Видео':<30}{'Метод':<7}{'Детект.,%':>11}{'Инлайеры':>10}{'FPS':>8}"
    print(f"\n{comparison_header}")
    print("-" * len(comparison_header))
    for entry in comparison_stats:
        for name in ("sift", "orb"):
            if name not in entry:
                continue
            metrics = entry[name]
            print(
                f"{entry['video_name'][:29]:<30}{name.upper():<7}"
                f"{metrics['detection_rate_pct']:>11.1f}{metrics['mean_inliers']:>10.1f}"
                f"{metrics['fps']:>8.1f}"
            )


def _parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки"""
    parser = argparse.ArgumentParser(description="Трекинг плоского объекта по ключевым точкам")
    parser.add_argument("--videos-dir", default="videos", help="каталог с видеофайлами")
    parser.add_argument("--output-dir", default="results", help="каталог для результатов")
    parser.add_argument(
        "--detector",
        default="sift",
        choices=("sift", "orb"),
        help="основной детектор признаков",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа: прогон по всем видео, сравнение детекторов и сохранение отчёта"""
    # Русский текст в консоли Windows иначе выводится в кодировке cp866
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = _parse_args()
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = _find_videos(videos_dir)
    if not video_files:
        print(f"Видеофайлы не найдены в каталоге: {videos_dir}")
        return

    primary_method = args.detector.lower()
    compare_method = "orb" if primary_method != "orb" else "sift"

    primary_stats: list[dict] = []
    comparison_stats: list[dict] = []

    for video_path in video_files:
        print(f"Обработка {video_path.name}: основной прогон ({primary_method.upper()})")
        primary = process_single_video(video_path, output_dir, method=primary_method, save_artifacts=True)
        primary_stats.append(primary)

        print(f"Обработка {video_path.name}: сравнительный прогон ({compare_method.upper()})")
        secondary = process_single_video(video_path, output_dir, method=compare_method, save_artifacts=False)
        comparison_stats.append(_build_comparison(video_path.name, primary, secondary))

    save_summary_charts(primary_stats, comparison_stats, output_dir)
    _save_metrics(primary_stats, comparison_stats, output_dir)
    _print_table(primary_stats, comparison_stats)


if __name__ == "__main__":
    main()
