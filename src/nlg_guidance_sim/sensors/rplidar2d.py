from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from nlg_guidance_sim.world.scene import Scene


def cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


@dataclass
class ScanResult:
    origin_xy: np.ndarray
    angles_rad: np.ndarray
    ranges_m: np.ndarray
    points_xy: np.ndarray
    hit_mask: np.ndarray

    def ordered_hit_points(self) -> np.ndarray:
        """Puntos con impacto, en el orden angular original del escáner."""
        return self.points_xy[self.hit_mask]


@dataclass
class RPLidar2DSim:
    """Simulador del Slamtec S2E (DTOF 360°).

    Defaults calibrados con la hoja de especificaciones oficial:
      - 32 000 muestras/s ÷ 10 Hz → 3 200 haces/vuelta
      - Resolución angular: 0.1125°
      - Alcance máximo (90 % reflectividad): 30 m
      - Precisión: ±30 mm  →  noise_std = 0.030 m
      - Rango ciego mínimo: 0.05 m
    """
    origin_x_m: float = 1.28
    origin_y_m: float = 0.0
    angle_min_deg: float = -60.0
    angle_max_deg: float = 60.0
    num_beams: int = 3200           # 32 000 pts/s ÷ 10 Hz
    max_range_m: float = 30.0       # 90 % reflectividad
    range_noise_std_m: float = 0.030  # ±30 mm
    seed: int = 7

    def origin(self) -> np.ndarray:
        return np.array([self.origin_x_m, self.origin_y_m], dtype=float)

    def beam_angles_rad(self) -> np.ndarray:
        return np.deg2rad(
            np.linspace(self.angle_min_deg, self.angle_max_deg, self.num_beams)
        )

    def _polygon_segments(
        self, polygon: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        segs: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            segs.append((p1, p2))
        return segs

    def _all_segments(
        self,
        scene: Scene,
        include_platform: bool = False,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        segments: list[tuple[np.ndarray, np.ndarray]] = []
        for poly in scene.scan_obstacle_polygons(include_platform=include_platform):
            segments.extend(self._polygon_segments(poly))
        return segments

    def _ray_segment_intersection(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
    ) -> tuple[bool, float, np.ndarray]:
        seg   = p2 - p1
        denom = cross2d(ray_dir, seg)
        if abs(denom) < 1e-9:
            return False, np.inf, np.array([np.nan, np.nan])

        diff = p1 - ray_origin
        t    = cross2d(diff, seg)     / denom
        u    = cross2d(diff, ray_dir) / denom

        if t >= 0.0 and 0.0 <= u <= 1.0:
            return True, float(t), ray_origin + t * ray_dir

        return False, np.inf, np.array([np.nan, np.nan])

    def scan(
        self,
        scene: Scene,
        include_platform: bool = False,
    ) -> ScanResult:
        """Ejecuta el ray casting sobre la escena.

        Parameters
        ----------
        scene:
            Escena 2D con geometría del NLG y la plataforma.
        include_platform:
            Propaga el flag a ``scene.scan_obstacle_polygons``.  Por defecto
            False: la plataforma (que aloja el sensor) no se detecta.
            Ponlo a True para simular la plataforma como clutter y aplicar
            filtrado en etapas posteriores del pipeline.
        """
        rng      = np.random.default_rng(self.seed)
        origin   = self.origin()
        angles   = self.beam_angles_rad()
        segments = self._all_segments(scene, include_platform=include_platform)

        ranges   = np.full_like(angles, fill_value=self.max_range_m, dtype=float)
        points   = np.zeros((len(angles), 2), dtype=float)
        hit_mask = np.zeros(len(angles), dtype=bool)

        for i, angle in enumerate(angles):
            direction  = np.array([math.cos(angle), math.sin(angle)], dtype=float)
            best_dist  = self.max_range_m
            best_point = origin + best_dist * direction
            hit        = False

            for p1, p2 in segments:
                ok, dist, point = self._ray_segment_intersection(
                    origin, direction, p1, p2
                )
                if ok and dist < best_dist:
                    best_dist  = dist
                    best_point = point
                    hit        = True

            if hit:
                noisy      = best_dist + rng.normal(0.0, self.range_noise_std_m)
                noisy      = float(np.clip(noisy, 0.0, self.max_range_m))
                best_point = origin + noisy * direction
                ranges[i]   = noisy
                points[i]   = best_point
                hit_mask[i] = True
            else:
                ranges[i]  = self.max_range_m
                points[i]  = best_point

        return ScanResult(
            origin_xy=origin,
            angles_rad=angles,
            ranges_m=ranges,
            points_xy=points,
            hit_mask=hit_mask,
        )
