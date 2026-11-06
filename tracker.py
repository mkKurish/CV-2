"""Трекинг плоского объекта по ключевым точкам с оценкой гомографии"""

import time
from dataclasses import dataclass

import cv2
import numpy as np

from detector import KeypointDetector, estimate_reference_corners

_GREEN = (0, 255, 0)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_MIN_SIDE = 7.0
_MAX_SIDE_RATIO = 2.4
_MIN_ANGLE_DEG = 25.0
_MAX_ANGLE_DEG = 155.0

# Ограничения для оптического сопровождения при размытии и отдалении
_MAX_COAST_FRAMES = 300
_MAX_FB_ERROR = 2.5
_MIN_FLOW_POINTS = 4
_ROI_PAD = 35
_ROI_UPSCALE = 2.5


@dataclass
class FrameTrackingResult:
    """Результат обработки одного кадра"""

    frame_idx: int
    detected: bool
    keypoints_count: int
    good_matches_count: int
    inliers_count: int
    inlier_ratio: float
    corners: np.ndarray | None
    homography: np.ndarray | None
    processing_time_ms: float
    good_matches: list[cv2.DMatch]
    inlier_mask: list[int] | None


class PlanarObjectTracker:
    """Сопоставляет эталонный кадр с текущим и отслеживает его границы"""

    def __init__(
        self,
        reference_frame: np.ndarray,
        label: str = "Tracked Object",
        method: str = "sift",
        max_features: int = 3000,
        ratio_thresh: float = 0.80,
        min_matches: int = 10,
        min_inliers: int = 10,
        ransac_thresh: float = 5.0,
        smoothing_alpha: float = 0.60,
    ) -> None:
        """Сохраняет параметры, эталон и вычисляет его признаки"""
        self.label = label
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.ransac_thresh = ransac_thresh
        self.smoothing_alpha = smoothing_alpha

        self.reference_frame = reference_frame.copy()
        self.detector = KeypointDetector(
            method=method,
            max_features=max_features,
            ratio_thresh=ratio_thresh,
        )

        # Ключевые точки и дескрипторы эталонного первого кадра
        self.ref_keypoints, self.ref_descriptors = (
            self.detector.detect_and_compute(self.reference_frame)
        )
        self.ref_corners = estimate_reference_corners(
            self.reference_frame, self.ref_keypoints
        )
        self.base_ref_keypoints_count = len(self.ref_keypoints)
        self.prev_corners: np.ndarray | None = None
        # Предыдущий серый кадр и счётчик кадров сопровождения потоком
        self.prev_gray: np.ndarray | None = None
        self.coast_frames: int = 0

    def track_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int = 0,
    ) -> tuple[np.ndarray, FrameTrackingResult]:
        """Обрабатывает кадр и возвращает размеченное изображение с метриками"""
        start = time.perf_counter()
        output = frame_bgr.copy()

        # Первый кадр — это сам эталон, поэтому трекинг начинается с известных углов
        if frame_idx == 0:
            corners = self.ref_corners.copy()
            self.prev_corners = corners.copy()
            self.prev_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            self.coast_frames = 0
            # Число точек базового масштаба — сопоставимо с остальными кадрами
            ref_count = self.base_ref_keypoints_count
            self._draw_overlay(
                output, corners, self.label, frame_idx, ref_count, ref_count, True
            )
            result = FrameTrackingResult(
                frame_idx=frame_idx,
                detected=True,
                keypoints_count=ref_count,
                good_matches_count=ref_count,
                inliers_count=ref_count,
                inlier_ratio=1.0,
                corners=corners,
                homography=np.eye(3, dtype=np.float64),
                processing_time_ms=(time.perf_counter() - start) * 1000.0,
                good_matches=[],
                inlier_mask=None,
            )
            return output, result

        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detect_and_compute(frame_bgr)
        good_matches = self.detector.match_descriptors(self.ref_descriptors, descriptors)

        corners: np.ndarray | None = None
        homography: np.ndarray | None = None
        inlier_mask: list[int] | None = None
        inliers_count = 0
        inlier_ratio = 0.0
        detected = False

        # Шаг А: прямое сопоставление дескрипторов полного кадра с эталоном
        if len(good_matches) >= self.min_matches:
            matrix, mask = self._find_homography(
                self.ref_keypoints, keypoints, good_matches
            )
            if matrix is not None and mask is not None:
                inlier_mask = mask
                inliers_count = int(sum(inlier_mask))
                inlier_ratio = inliers_count / len(good_matches)
                proj_corners = cv2.perspectiveTransform(self.ref_corners, matrix)
                quad_valid = self._is_valid_quad(proj_corners, frame_bgr.shape)
                consistent = self._is_consistent_with_prev(
                    proj_corners, frame_bgr.shape, inliers_count, inlier_ratio
                )

                if inliers_count >= self.min_inliers and quad_valid and consistent:
                    corners = self._smooth_corners(proj_corners, frame_bgr.shape)
                    self.prev_corners = corners.copy()
                    homography = matrix
                    detected = True
                    self.coast_frames = 0

        # Шаг Б: локальный апскейл вокруг рамки, когда объект стал мелким
        if not detected and self.prev_corners is not None:
            roi_box = self._expanded_roi(self.prev_corners, frame_bgr.shape)
            roi_w = roi_box[2] - roi_box[0]
            roi_h = roi_box[3] - roi_box[1]
            if max(roi_w, roi_h) <= 260:
                roi_keypoints, roi_descriptors = self.detector.detect_in_upscaled_roi(
                    frame_bgr, roi_box, scale=_ROI_UPSCALE
                )
                roi_matches = self.detector.match_descriptors(
                    self.ref_descriptors, roi_descriptors
                )
                if len(roi_matches) >= self.min_matches:
                    matrix, mask = self._find_homography(
                        self.ref_keypoints, roi_keypoints, roi_matches
                    )
                    if matrix is not None and mask is not None:
                        roi_inliers = int(sum(mask))
                        roi_ratio = roi_inliers / len(roi_matches)
                        proj_corners = cv2.perspectiveTransform(self.ref_corners, matrix)
                        if (
                            roi_inliers >= self.min_inliers
                            and self._is_valid_quad(proj_corners, frame_bgr.shape)
                            and self._is_consistent_with_prev(
                                proj_corners, frame_bgr.shape, roi_inliers, roi_ratio
                            )
                        ):
                            corners = self._smooth_corners(proj_corners, frame_bgr.shape)
                            self.prev_corners = corners.copy()
                            homography = matrix
                            inlier_mask = mask
                            inliers_count = roi_inliers
                            inlier_ratio = roi_ratio
                            good_matches = roi_matches
                            detected = True
                            self.coast_frames = 0

        # Шаг В: оптический поток по последней надёжной рамке
        if (
            not detected
            and self.prev_corners is not None
            and self.prev_gray is not None
            and self.coast_frames < _MAX_COAST_FRAMES
        ):
            proj_corners, flow_inliers, flow_total = self._track_by_optical_flow(
                self.prev_gray, curr_gray, self.prev_corners
            )
            if (
                proj_corners is not None
                and self._is_valid_quad(proj_corners, frame_bgr.shape)
                and self._is_consistent_with_prev(proj_corners, frame_bgr.shape, 8, 0.25)
            ):
                corners = self._smooth_corners(proj_corners, frame_bgr.shape)
                self.prev_corners = corners.copy()
                self.coast_frames += 1
                detected = True
                if inliers_count < self.min_inliers:
                    inliers_count = min(flow_inliers, 15)
                    inlier_ratio = flow_inliers / flow_total if flow_total else 0.0

        if not detected:
            self.prev_corners = None
            corners = None

        # Предыдущий серый кадр нужен для оптического потока на следующем шаге
        self.prev_gray = curr_gray

        self._draw_overlay(
            output,
            corners,
            self.label,
            frame_idx,
            inliers_count,
            len(good_matches),
            detected,
        )
        result = FrameTrackingResult(
            frame_idx=frame_idx,
            detected=detected,
            keypoints_count=len(keypoints),
            good_matches_count=len(good_matches),
            inliers_count=inliers_count,
            inlier_ratio=inlier_ratio,
            corners=corners,
            homography=homography,
            processing_time_ms=(time.perf_counter() - start) * 1000.0,
            good_matches=good_matches,
            inlier_mask=inlier_mask,
        )
        return output, result

    def _find_homography(
        self,
        ref_keypoints: list[cv2.KeyPoint],
        frame_keypoints: list[cv2.KeyPoint],
        good_matches: list[cv2.DMatch],
    ) -> tuple[np.ndarray | None, list[int] | None]:
        """Оценивает гомографию по совпадениям и возвращает маску инлайеров"""
        if len(good_matches) < 4:
            return None, None

        src_pts = np.float32(
            [ref_keypoints[match.queryIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [frame_keypoints[match.trainIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            self.ransac_thresh,
            maxIters=1000,
            confidence=0.99,
        )
        if matrix is None or mask is None:
            return None, None
        return matrix, mask.ravel().astype(int).tolist()

    def _expanded_roi(
        self,
        corners: np.ndarray,
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        """Возвращает прямоугольник вокруг рамки с запасом, обрезанный по кадру"""
        pts = corners.reshape(-1, 2)
        h, w = frame_shape[:2]
        x0 = int(max(0, np.floor(pts[:, 0].min()) - _ROI_PAD))
        y0 = int(max(0, np.floor(pts[:, 1].min()) - _ROI_PAD))
        x1 = int(min(w, np.ceil(pts[:, 0].max()) + _ROI_PAD))
        y1 = int(min(h, np.ceil(pts[:, 1].max()) + _ROI_PAD))
        return x0, y0, x1, y1

    def _inner_grid_points(
        self,
        corners: np.ndarray,
        grid: int = 4,
    ) -> list[np.ndarray]:
        """Строит сетку внутренних точек четырёхугольника для оптического потока"""
        src = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        dst = corners.reshape(4, 2).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        fractions = np.linspace(0.2, 0.8, grid)
        grid_pts = np.float32(
            [[x, y] for y in fractions for x in fractions]
        ).reshape(-1, 1, 2)
        return list(cv2.perspectiveTransform(grid_pts, matrix).reshape(-1, 2))

    def _track_by_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_corners: np.ndarray,
    ) -> tuple[np.ndarray | None, int, int]:
        """Смещает рамку по оптическому потоку.

        Возвращает спроецированные углы, число инлайеров и число
        отслеженных точек. Аффинная модель устойчивее гомографии
        на малых и размытых объектах.
        """
        pts = prev_corners.reshape(4, 2).astype(np.float32)
        center = pts.mean(axis=0, keepdims=True)
        shrunk_pts = center + 0.90 * (pts - center)
        mask = np.zeros_like(prev_gray)
        cv2.fillConvexPoly(mask, np.int32(shrunk_pts).reshape(-1, 1, 2), 255)

        features = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=4,
            mask=mask,
        )
        points = [] if features is None else list(features.reshape(-1, 2))
        # Всегда добавляем внутреннюю регулярную сетку для устойчивости на малом масштабе
        points.extend(self._inner_grid_points(prev_corners, grid=5))
        if len(points) < _MIN_FLOW_POINTS:
            return None, 0, len(points)

        p0 = np.float32(points).reshape(-1, 1, 2)
        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, p0, None, winSize=(21, 21), maxLevel=3
        )
        if p1 is None or status is None:
            return None, 0, len(points)

        # Обратный поток отсеивает точки, ушедшие на другой объект
        p0_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, p1, None, winSize=(21, 21), maxLevel=3
        )
        if p0_back is None or status_back is None:
            return None, 0, len(points)

        forward_ok = status.ravel().astype(bool) & status_back.ravel().astype(bool)
        fb_error = np.linalg.norm(
            p0_back.reshape(-1, 2) - p0.reshape(-1, 2), axis=1
        )
        keep = forward_ok & (fb_error < _MAX_FB_ERROR)
        p0_good = p0.reshape(-1, 2)[keep]
        p1_good = p1.reshape(-1, 2)[keep]
        if len(p0_good) < _MIN_FLOW_POINTS:
            return None, 0, len(p0_good)

        affine, inliers = cv2.estimateAffinePartial2D(
            p0_good,
            p1_good,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if affine is None or inliers is None:
            return None, 0, len(p0_good)

        delta = np.vstack([affine, [0.0, 0.0, 1.0]])
        proj_corners = cv2.perspectiveTransform(prev_corners, delta)
        return proj_corners, int(inliers.sum()), len(p0_good)

    def _smooth_corners(
        self,
        corners: np.ndarray,
        frame_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Сглаживает углы, если они не сместились скачком"""
        if self.prev_corners is None:
            return corners

        h, w = frame_shape[:2]
        shift = float(np.mean(np.linalg.norm(corners - self.prev_corners, axis=2)))
        if shift >= 0.25 * max(h, w):
            return corners

        alpha = self.smoothing_alpha
        return alpha * corners + (1.0 - alpha) * self.prev_corners

    def _is_consistent_with_prev(
        self,
        corners: np.ndarray,
        frame_shape: tuple[int, ...],
        inliers_count: int,
        inlier_ratio: float = 0.0,
    ) -> bool:
        """Отсекает ложные выбросы гомографии, резко меняющие площадь или позицию"""
        h, w = frame_shape[:2]
        frame_area = float(h * w)
        pts = corners.reshape(4, 2).astype(np.float32)
        area = float(cv2.contourArea(pts))

        if self.prev_corners is None:
            return (
                inliers_count >= max(10, self.min_inliers)
                and inlier_ratio >= 0.25
                and area <= 0.75 * frame_area
            )

        # Уверенное прямое переобнаружение по большому числу и высокой доле инлайеров
        if inliers_count >= 25 and inlier_ratio >= 0.40 and area <= 0.75 * frame_area:
            return True

        prev_pts = self.prev_corners.reshape(4, 2).astype(np.float32)
        prev_area = float(cv2.contourArea(prev_pts))
        if prev_area <= 1.0:
            return True

        area_ratio = area / prev_area
        if not 0.55 <= area_ratio <= 1.75:
            return False

        center_shift = float(np.linalg.norm(pts.mean(axis=0) - prev_pts.mean(axis=0)))
        if center_shift > 0.16 * max(h, w):
            return False

        return True

    def _is_valid_quad(self, corners: np.ndarray, frame_shape: tuple[int, ...]) -> bool:
        """Проверяет, что спроецированный четырёхугольник правдоподобен"""
        if not np.all(np.isfinite(corners)):
            return False

        pts = corners.reshape(4, 2).astype(np.float32)
        if not cv2.isContourConvex(pts.astype(np.int32)):
            return False

        frame_area = float(frame_shape[0] * frame_shape[1])
        area = cv2.contourArea(pts)
        if not 0.001 * frame_area <= area <= 1.8 * frame_area:
            return False

        # Стороны: длины и соотношение противоположных сторон
        sides = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        if np.any(sides < _MIN_SIDE):
            return False
        for first, second in ((0, 2), (1, 3)):
            ratio = max(sides[first], sides[second]) / min(sides[first], sides[second])
            if ratio > _MAX_SIDE_RATIO:
                return False

        # Внутренние углы через косинус между смежными сторонами
        for index in range(4):
            current = pts[index]
            prev_vec = pts[index - 1] - current
            next_vec = pts[(index + 1) % 4] - current
            norm = np.linalg.norm(prev_vec) * np.linalg.norm(next_vec)
            if norm == 0.0:
                return False
            cosine = float(np.dot(prev_vec, next_vec) / norm)
            angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
            if not _MIN_ANGLE_DEG <= angle <= _MAX_ANGLE_DEG:
                return False

        return True

    def _draw_overlay(
        self,
        frame: np.ndarray,
        corners: np.ndarray | None,
        label: str,
        frame_idx: int,
        inliers: int,
        matches: int,
        detected: bool,
    ) -> None:
        """Рисует рамку объекта и статусную панель"""
        if detected and corners is not None:
            pts = np.int32(corners).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], True, _GREEN, 3)
            for point in pts.reshape(-1, 2):
                cv2.circle(frame, tuple(int(value) for value in point), 5, _GREEN, -1)

            # Подпись над самой верхней вершиной рамки, не выходя за кадр
            flat = pts.reshape(-1, 2)
            top = flat[int(np.argmin(flat[:, 1]))]
            self._draw_label(frame, label, int(top[0]), int(top[1]), inliers)

        status = (
            f"Frame: {frame_idx} | Status: {'TRACKED' if detected else 'LOST'}"
            f" | Matches: {matches} | Inliers: {inliers}"
        )
        self._draw_status(frame, status, detected)

    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        x: int,
        y: int,
        inliers: int,
    ) -> None:
        """Рисует залитую плашку с подписью объекта"""
        text = f"{label} | inliers: {inliers}"
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)

        h, w = frame.shape[:2]
        pad = 6
        x0 = int(np.clip(x - text_w // 2, 0, max(0, w - text_w - 2 * pad)))
        y0 = int(np.clip(y - text_h - 2 * pad - 10, 0, max(0, h - text_h - 2 * pad)))
        cv2.rectangle(
            frame,
            (x0, y0),
            (x0 + text_w + 2 * pad, y0 + text_h + 2 * pad),
            _GREEN,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x0 + pad, y0 + pad + text_h),
            font,
            scale,
            _BLACK,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_status(self, frame: np.ndarray, text: str, detected: bool) -> None:
        """Рисует компактную тёмную плашку со статусом в левом нижнем углу"""
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

        pad = 8
        h = frame.shape[0]
        # Нижний угол не пересекается с подписью объекта в верхней части кадра
        x0 = 10
        y0 = h - text_h - baseline - 2 * pad - 10
        x1, y1 = x0 + text_w + 2 * pad, y0 + text_h + baseline + 2 * pad

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), _BLACK, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0, frame)
        cv2.putText(
            frame,
            text,
            (x0 + pad, y0 + pad + text_h),
            font,
            scale,
            _GREEN if detected else _WHITE,
            thickness,
            cv2.LINE_AA,
        )
