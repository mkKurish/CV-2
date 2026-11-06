"""Детектирование ключевых точек и сопоставление дескрипторов (SIFT/ORB)."""

import cv2
import numpy as np

# Минимум точек, чтобы рамка по перцентилям была осмысленной
_MIN_KEYPOINTS_FOR_BBOX = 20


class KeypointDetector:
    """Обёртка над детектором признаков и BFMatcher с критерием Лоу."""

    def __init__(
        self,
        method: str = "sift",
        max_features: int = 3000,
        ratio_thresh: float = 0.80,
    ) -> None:
        """Создаёт детектор и матчер для выбранного метода."""
        self.method = method.lower()
        self.max_features = max_features
        self.ratio_thresh = ratio_thresh

        if self.method == "sift":
            # Низкий contrastThreshold помогает на размытых малоконтрастных участках
            self.detector = cv2.SIFT_create(
                nfeatures=max_features,
                nOctaveLayers=4,
                contrastThreshold=0.02,
                edgeThreshold=12,
                sigma=1.4,
            )
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif self.method == "orb":
            # Мелкий scaleFactor и больше уровней — точки на сильно уменьшенном объекте
            self.detector = cv2.ORB_create(
                nfeatures=max_features,
                scaleFactor=1.15,
                nlevels=12,
                edgeThreshold=15,
                fastThreshold=10,
            )
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"Неизвестный метод детектора: {method!r}")

    def detect_and_compute(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        """Находит ключевые точки и дескрипторы кадра."""
        gray = (
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if frame_bgr.ndim == 3
            else frame_bgr
        )
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return list(keypoints), descriptors

    def detect_in_upscaled_roi(
        self,
        frame_bgr: np.ndarray,
        roi_box: tuple[int, int, int, int],
        scale: float = 2.0,
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        """Ищет признаки в увеличенной области и возвращает точки в координатах кадра.

        Локальный апскейл помогает, когда объект сильно отдалился: детали
        становятся крупнее, а координаты пересчитываются обратно.
        """
        x0, y0, x1, y1 = roi_box
        h, w = frame_bgr.shape[:2]
        x0 = int(np.clip(x0, 0, w))
        x1 = int(np.clip(x1, 0, w))
        y0 = int(np.clip(y0, 0, h))
        y1 = int(np.clip(y1, 0, h))

        if x1 - x0 < 2 or y1 - y0 < 2 or scale <= 0.0:
            return [], None

        roi = frame_bgr[y0:y1, x0:x1]
        upscaled = cv2.resize(
            roi,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        keypoints, descriptors = self.detect_and_compute(upscaled)

        # Координаты точек возвращаются в систему координат исходного кадра
        mapped = [
            cv2.KeyPoint(
                x=x0 + kp.pt[0] / scale,
                y=y0 + kp.pt[1] / scale,
                size=kp.size / scale,
                angle=kp.angle,
                response=kp.response,
                octave=kp.octave,
                class_id=kp.class_id,
            )
            for kp in keypoints
        ]
        return mapped, descriptors

    def match_descriptors(
        self,
        desc_ref: np.ndarray | None,
        desc_frame: np.ndarray | None,
    ) -> list[cv2.DMatch]:
        """Сопоставляет дескрипторы методом kNN с критерием Лоу и дедупликацией."""
        if desc_ref is None or desc_frame is None:
            return []
        if len(desc_ref) < 2 or len(desc_frame) < 2:
            return []

        raw_matches = self.matcher.knnMatch(desc_ref, desc_frame, k=2)
        good_matches = [
            pair[0]
            for pair in raw_matches
            if len(pair) == 2
            and pair[0].distance < self.ratio_thresh * pair[1].distance
        ]
        good_matches.sort(key=lambda match: match.distance)

        # Дедупликация: на точку кадра и на точку эталона — только лучшее совпадение
        unique_matches: list[cv2.DMatch] = []
        seen_train: set[int] = set()
        seen_query: set[int] = set()
        for match in good_matches:
            if match.trainIdx in seen_train or match.queryIdx in seen_query:
                continue
            seen_train.add(match.trainIdx)
            seen_query.add(match.queryIdx)
            unique_matches.append(match)
        return unique_matches


def estimate_reference_corners(
    frame_bgr: np.ndarray,
    keypoints: list[cv2.KeyPoint] | None = None,
) -> np.ndarray:
    """Оценивает углы эталонного объекта на первом кадре."""
    h, w = frame_bgr.shape[:2]
    full_corners = np.float32(
        [[2, 2], [w - 3, 2], [w - 3, h - 3], [2, h - 3]]
    ).reshape(4, 1, 2)

    if not keypoints or len(keypoints) < _MIN_KEYPOINTS_FOR_BBOX:
        return full_corners

    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    x_min, x_max = np.percentile(pts[:, 0], [2, 98])
    y_min, y_max = np.percentile(pts[:, 1], [2, 98])

    # Небольшой запас, чтобы точки не лежали ровно на границе рамки
    pad_x = 0.03 * (x_max - x_min)
    pad_y = 0.03 * (y_max - y_min)
    x_min = float(np.clip(x_min - pad_x, 2, w - 3))
    x_max = float(np.clip(x_max + pad_x, 2, w - 3))
    y_min = float(np.clip(y_min - pad_y, 2, h - 3))
    y_max = float(np.clip(y_max + pad_y, 2, h - 3))

    area = (x_max - x_min) * (y_max - y_min)
    frame_area = float(w * h)
    if 0.10 * frame_area < area < 0.75 * frame_area:
        return np.float32(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        ).reshape(4, 1, 2)

    return full_corners
