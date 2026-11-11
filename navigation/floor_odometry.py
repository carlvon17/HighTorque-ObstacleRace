"""Metric dense floor-texture motion using observed depth and IMU attitude."""
import cv2
import numpy as np


class FloorOdometry:
    """Register a small measured floor patch in the previous camera heading.

    A displacement is accepted only when texture has two observable directions,
    the NCC peak is distinct, and phase correlation agrees with that peak.
    """

    def __init__(self, intrinsics, camera_position, camera_rotation):
        self.intrinsics = tuple(intrinsics)
        self.position = np.asarray(camera_position, dtype=float)
        self.rotation = np.asarray(camera_rotation, dtype=float)
        self.resolution = .002
        self.x, self.y = np.meshgrid(np.arange(.45, .85, self.resolution),
                                     np.arange(-.10, .10, self.resolution), indexing='ij')
        self.support = 0
        self.reason = 'initializing'
        self.quality = {}

    def project(self, gray, depth, attitude, plane, heading):
        if plane is None or not plane.valid:
            return None
        cosine, sine = np.cos(heading), np.sin(heading)
        x = cosine * self.x - sine * self.y
        y = sine * self.x + cosine * self.y
        z = -(plane.normal[0] * x + plane.normal[1] * y + plane.offset) / plane.normal[2]
        level = np.stack([x, y, z], axis=-1)
        optical = (level @ attitude - self.position) @ self.rotation
        fx, fy, cx, cy = self.intrinsics
        map_x = np.asarray(fx * optical[..., 0] / optical[..., 2] + cx, np.float32)
        map_y = np.asarray(fy * optical[..., 1] / optical[..., 2] + cy, np.float32)
        height, width = gray.shape
        # No padded or extrapolated image pixel contributes to the measurement.
        inside = ((optical[..., 2] > .15) & (map_x >= 1.) & (map_x < width - 2.) &
                  (map_y >= 1.) & (map_y < height - 2.))
        if not inside.all():
            return None
        # Bilinear gray sampling uses four source pixels. Each source must
        # measure the floor along its own camera ray, including depth holes
        # and obstacle edges that a nearest-neighbor depth check would miss.
        columns = np.floor(map_x).astype(np.int32)
        rows = np.floor(map_y).astype(np.int32)
        optical_normal = plane.normal @ attitude @ self.rotation
        optical_offset = plane.offset + (self.position @ attitude.T) @ plane.normal
        for row_offset, column_offset in ((0, 0), (0, 1), (1, 0), (1, 1)):
            source_rows = rows + row_offset
            source_columns = columns + column_offset
            ray_dot_normal = (optical_normal[0] * (source_columns - cx) / fx +
                              optical_normal[1] * (source_rows - cy) / fy + optical_normal[2])
            with np.errstate(divide='ignore', invalid='ignore'):
                expected_depth = -optical_offset / ray_dot_normal
            measured_depth = depth[source_rows, source_columns]
            on_floor = (np.isfinite(measured_depth) & np.isfinite(expected_depth) &
                        (expected_depth > .15) & (abs(measured_depth - expected_depth) < .015))
            if not on_floor.all():
                return None
        image = cv2.remap(gray.astype(np.float32), map_x, map_y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        texture = image - cv2.GaussianBlur(image, (0, 0), 2.)
        # White-line edges have high contrast but leave motion along the line
        # ambiguous. Bound their contribution so measured carpet grain remains
        # visible to the unchanged texture, peak and phase-consistency checks.
        limit = max(.25, 3 * np.median(abs(texture - np.median(texture))))
        return np.ascontiguousarray(np.clip(texture, -limit, limit))

    @staticmethod
    def texture_quality(image):
        dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.column_stack([dx.ravel(), dy.ravel()])
        eigenvalues = np.linalg.eigvalsh(gradient.T @ gradient / len(gradient))
        ratio = float(eigenvalues[0] / max(eigenvalues[1], 1e-12))
        return float(np.std(image)), ratio

    def match(self, reference, current, max_distance):
        """Return forward/lateral displacement in the reference heading frame."""
        self.support = 0
        self.quality = {}
        std_old, ratio_old = self.texture_quality(reference)
        std_new, ratio_new = self.texture_quality(current)
        self.quality.update(texture_std=min(std_old, std_new),
                            direction_ratio=min(ratio_old, ratio_new))
        if min(std_old, std_new) < .10 or min(ratio_old, ratio_new) < .01:
            self.reason = 'insufficient_floor_texture'
            return None
        margin = max(3, int(np.ceil(max_distance / self.resolution)))
        if min(current.shape) <= 2 * margin + 12:
            self.reason = 'insufficient_search_overlap'
            return None
        template = current[margin:-margin, margin:-margin]
        correlation = cv2.matchTemplate(reference, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, peak = cv2.minMaxLoc(correlation)
        px, py = peak
        height, width = correlation.shape
        if px <= 1 or py <= 1 or px >= width - 2 or py >= height - 2:
            self.reason = 'floor_motion_outside_search'
            return None
        other = correlation.copy()
        other[max(0, py - 4):py + 5, max(0, px - 4):px + 5] = -1.
        separation = score - float(np.max(other))
        self.quality.update(ncc=float(score), peak_separation=separation)
        if score < .45 or separation < .03:
            self.reason = 'ambiguous_floor_match'
            return None
        subpixel = []
        for left, center, right in ((correlation[py, px - 1], correlation[py, px], correlation[py, px + 1]),
                                    (correlation[py - 1, px], correlation[py, px], correlation[py + 1, px])):
            denominator = left - 2 * center + right
            offset = .5 * (left - right) / denominator if abs(denominator) > 1e-8 else 0.
            subpixel.append(np.clip(offset, -.5, .5))
        displacement = (np.array([py + subpixel[1], px + subpixel[0]]) - margin) * self.resolution
        if np.linalg.norm(displacement) > max_distance:
            self.reason = 'floor_motion_outside_search'
            return None
        window = cv2.createHanningWindow((reference.shape[1], reference.shape[0]), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(reference.copy(), current.copy(), window)
        phase_displacement = -np.array([shift[1], shift[0]]) * self.resolution
        disagreement = float(np.linalg.norm(displacement - phase_displacement))
        self.quality.update(phase_response=float(response), agreement_m=disagreement)
        if response < .35 or disagreement > .006:
            self.reason = 'inconsistent_floor_match'
            return None
        self.support = int(template.size)
        self.reason = 'measured_floor_texture'
        return displacement

    def estimate(self, old_gray, old_depth, old_attitude, old_plane,
                 gray, depth, attitude, plane, elapsed):
        self.support = 0
        self.quality = {}
        if not np.isfinite(elapsed) or elapsed <= 0:
            self.reason = 'invalid_floor_interval'
            return None
        heading = float(np.arctan2(old_attitude[1, 0], old_attitude[0, 0]))
        reference = self.project(old_gray, old_depth, old_attitude, old_plane, heading)
        current = self.project(gray, depth, attitude, plane, heading)
        if reference is None or current is None:
            self.reason = 'floor_patch_not_observed'
            return None
        # A stopped camera outage need not discard a still-visible metric
        # patch. Reacquire its measured displacement within a bounded search;
        # missing or ambiguous texture still rejects the entire update.
        max_distance = min(.06, .6 * elapsed + .01) if elapsed <= .25 else .08
        displacement = self.match(reference, current, max_distance)
        if displacement is None:
            return None
        cosine, sine = np.cos(heading), np.sin(heading)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        return displacement @ rotation.T
