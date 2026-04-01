import csv
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from matcher import LightGlueMatcher


ACCEPTED_PAIR_CACHE_HEADER = [
    'pair_id',
    'kf_prev_id',
    'kf_curr_id',
    'src_prev_id',
    'src_curr_id',
    'image_prev',
    'image_curr',
    'timestamp_prev',
    'timestamp_curr',

    'odom_translation_m',
    'odom_rotation_deg',

    'num_keypoints0',
    'num_keypoints1',
    'num_matches',
    'num_inliers',
    'match_inlier_ratio',
    'mean_match_score',

    'coverage0',
    'coverage1',
    'parallax_mean_px',
    'parallax_median_px',

    'geo_error_mean',
    'geo_error_median',

    'vis_pose_ok',
    'vis_rot_deg',
    'e_rot_iv_deg',
    'e_trans_dir_iv_deg',
]


def compute_translation(row0, row1):
    required_keys = ['pos_x', 'pos_y', 'pos_z']
    if not all(k in row0 for k in required_keys):
        return None
    if not all(k in row1 for k in required_keys):
        return None

    dx = row1['pos_x'] - row0['pos_x']
    dy = row1['pos_y'] - row0['pos_y']
    dz = row1['pos_z'] - row0['pos_z']
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def quat_xyzw_to_rotmat(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError('invalid quaternion norm')

    qx, qy, qz, qw = q / n
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw),       2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw),       1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw),       2.0 * (qy * qz + qx * qw),       1.0 - 2.0 * (qx * qx + qy * qy)]
    ], dtype=np.float64)


def compute_rotation_deg(row0, row1):
    required_keys = ['quat_x', 'quat_y', 'quat_z', 'quat_w']
    if not all(k in row0 for k in required_keys):
        return None
    if not all(k in row1 for k in required_keys):
        return None

    try:
        R0 = quat_xyzw_to_rotmat(row0['quat_x'], row0['quat_y'], row0['quat_z'], row0['quat_w'])
        R1 = quat_xyzw_to_rotmat(row1['quat_x'], row1['quat_y'], row1['quat_z'], row1['quat_w'])
    except Exception:
        return None

    R_rel = R0.T @ R1
    trace_val = np.trace(R_rel)
    c = float(np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(c))


def load_camera_matrix(camera_info_path: Path, image_shape=None):
    if camera_info_path.exists():
        with open(camera_info_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if 'camera_matrix' in data and 'data' in data['camera_matrix']:
            k = data['camera_matrix']['data']
            K = np.array(k, dtype=np.float64).reshape(3, 3)
            return K

    if image_shape is None:
        raise FileNotFoundError(f'找不到 camera_info.yaml，也沒有 image_shape 可建立近似內參: {camera_info_path}')

    h, w = image_shape[:2]
    fx = float(max(h, w))
    fy = float(max(h, w))
    cx = float(w) / 2.0
    cy = float(h) / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    return K


def compute_row_rotation_matrix(row):
    required_keys = ['quat_x', 'quat_y', 'quat_z', 'quat_w']
    if not all(k in row for k in required_keys):
        return None

    try:
        return quat_xyzw_to_rotmat(
            row['quat_x'],
            row['quat_y'],
            row['quat_z'],
            row['quat_w']
        )
    except Exception:
        return None


def compute_translation_from_rows(row0, row1):
    required_keys = ['pos_x', 'pos_y', 'pos_z']
    if not all(k in row0 for k in required_keys):
        return None
    if not all(k in row1 for k in required_keys):
        return None

    return np.array([
        row1['pos_x'] - row0['pos_x'],
        row1['pos_y'] - row0['pos_y'],
        row1['pos_z'] - row0['pos_z']
    ], dtype=np.float64)


def compute_odom_relative_pose(row0, row1):
    R0 = compute_row_rotation_matrix(row0)
    R1 = compute_row_rotation_matrix(row1)
    t_w = compute_translation_from_rows(row0, row1)

    if R0 is None or R1 is None or t_w is None:
        return None, None

    R_rel = R0.T @ R1
    t_rel_in_k = R0.T @ t_w
    return R_rel, t_rel_in_k


def rotation_angle_deg(R):
    trace_val = np.trace(R)
    c = (trace_val - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return math.degrees(math.acos(c))


def rotation_distance_deg(Ra, Rb):
    return rotation_angle_deg(Ra.T @ Rb)


def vector_angle_deg(v0, v1):
    n0 = np.linalg.norm(v0)
    n1 = np.linalg.norm(v1)
    if n0 < 1e-12 or n1 < 1e-12:
        return -1.0

    c = float(np.dot(v0, v1) / (n0 * n1))
    c = float(np.clip(c, -1.0, 1.0))
    return math.degrees(math.acos(c))


def compute_coverage(points, width, height, rows=4, cols=4):
    if len(points) == 0:
        return 0.0

    occupied = np.zeros((rows, cols), dtype=np.uint8)

    for x, y in points:
        c = min(cols - 1, max(0, int(x / max(width, 1) * cols)))
        r = min(rows - 1, max(0, int(y / max(height, 1) * rows)))
        occupied[r, c] = 1

    return float(occupied.sum()) / float(rows * cols)


def compute_parallax_stats(pts0, pts1):
    if len(pts0) == 0:
        return 0.0, 0.0
    d = np.linalg.norm(pts1 - pts0, axis=1)
    return float(d.mean()), float(np.median(d))


def compute_sampson_error(F, pts0, pts1):
    if F is None or len(pts0) == 0:
        return np.array([], dtype=np.float64)

    pts0_h = np.hstack([pts0, np.ones((len(pts0), 1), dtype=np.float64)])
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1), dtype=np.float64)])

    Fx1 = (F @ pts0_h.T).T
    Ftx2 = (F.T @ pts1_h.T).T
    x2tFx1 = np.sum(pts1_h * Fx1, axis=1)

    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    denom = np.clip(denom, 1e-12, None)
    err = (x2tFx1 ** 2) / denom
    return err.astype(np.float64)


def normalize_t_if_possible(t):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(t)
    if n < 1e-12:
        return t
    return t / n


def extract_first_essential_matrix(E):
    if E is None:
        return None
    E = np.asarray(E, dtype=np.float64)
    if E.shape == (3, 3):
        return E
    if E.ndim == 2 and E.shape[1] == 3 and E.shape[0] % 3 == 0:
        return E[:3, :]
    return None


def estimate_visual_geometry(pts0, pts1, K, min_matches_for_geometry=8):
    result = {
        'pose_ok': False,
        'R_vis': np.eye(3, dtype=np.float64),
        't_vis_unit': np.zeros((3,), dtype=np.float64),
        'num_inliers': 0,
        'inlier_ratio': 0.0,
        'pose_mask': np.zeros((len(pts0),), dtype=bool),
        'geo_error_mean': -1.0,
        'geo_error_median': -1.0,
        'vis_rot_deg': -1.0,
    }

    if len(pts0) < min_matches_for_geometry:
        return result

    F, _ = cv2.findFundamentalMat(
        pts0,
        pts1,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.999
    )

    E, mask_e = cv2.findEssentialMat(
        pts0,
        pts1,
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    E = extract_first_essential_matrix(E)
    if E is None or mask_e is None:
        return result

    mask_e = mask_e.reshape(-1).astype(bool)
    if int(mask_e.sum()) < min_matches_for_geometry:
        return result

    _, R, t, mask_pose = cv2.recoverPose(E, pts0, pts1, cameraMatrix=K, mask=mask_e.astype(np.uint8))
    if mask_pose is None:
        return result

    pose_mask = mask_pose.reshape(-1).astype(bool)
    num_inliers = int(pose_mask.sum())
    inlier_ratio = float(num_inliers) / float(len(pts0)) if len(pts0) > 0 else 0.0

    inlier_pts0 = pts0[pose_mask]
    inlier_pts1 = pts1[pose_mask]

    sampson = compute_sampson_error(F, inlier_pts0, inlier_pts1)
    geo_error_mean = float(sampson.mean()) if len(sampson) > 0 else -1.0
    geo_error_median = float(np.median(sampson)) if len(sampson) > 0 else -1.0

    vis_rot_deg = rotation_angle_deg(R)

    result.update({
        'pose_ok': True,
        'R_vis': R,
        't_vis_unit': normalize_t_if_possible(t.reshape(3)),
        'num_inliers': num_inliers,
        'inlier_ratio': inlier_ratio,
        'pose_mask': pose_mask,
        'geo_error_mean': geo_error_mean,
        'geo_error_median': geo_error_median,
        'vis_rot_deg': vis_rot_deg,
    })
    return result


class KeyframeSelector:
    def __init__(
        self,
        prefilter_translation_m=0.20,
        prefilter_rotation_deg=2.0,
        min_matches=120,
        min_inlier_ratio=0.30,
        min_translation_m=0.50,
        min_rotation_deg=5.0,
        matcher=None,
        cache_csv_path='',
        camera_info_path='',
        grid_rows=4,
        grid_cols=4,
        min_matches_for_geometry=8,
    ):
        self.prefilter_translation_m = prefilter_translation_m
        self.prefilter_rotation_deg = prefilter_rotation_deg
        self.min_matches = min_matches
        self.min_inlier_ratio = min_inlier_ratio
        self.min_translation_m = min_translation_m
        self.min_rotation_deg = min_rotation_deg

        self.matcher = matcher or LightGlueMatcher()

        self.cache_csv_path = Path(cache_csv_path).expanduser().resolve() if cache_csv_path else None
        self.camera_info_path = Path(camera_info_path).expanduser().resolve() if camera_info_path else None
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.min_matches_for_geometry = int(min_matches_for_geometry)

        self.last_kf_row = None
        self.last_kf_img = None
        self.last_kf_feats = None
        self.keyframe_id = 0
        self.K = None

        if self.cache_csv_path is not None:
            self._init_cache_file()

    def _init_cache_file(self):
        self.cache_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ACCEPTED_PAIR_CACHE_HEADER)
            writer.writeheader()

    def _append_cache_row(self, cache_row):
        if self.cache_csv_path is None:
            return

        with open(self.cache_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ACCEPTED_PAIR_CACHE_HEADER)
            writer.writerow(cache_row)

    def _ensure_camera_matrix(self, img_shape):
        if self.K is not None:
            return self.K

        if self.camera_info_path is None:
            self.K = load_camera_matrix(Path(''), image_shape=img_shape)
        else:
            self.K = load_camera_matrix(self.camera_info_path, image_shape=img_shape)

        return self.K

    def _build_accepted_pair_cache(
        self,
        prev_row,
        curr_row,
        prev_img,
        curr_img,
        match_result,
        translation_m,
        rotation_deg,
        kf_prev_id,
        kf_curr_id,
    ):
        K = self._ensure_camera_matrix(prev_img.shape)

        pts0 = match_result['mkpts0']
        pts1 = match_result['mkpts1']
        match_scores = match_result['match_scores']

        geo = estimate_visual_geometry(
            pts0,
            pts1,
            K,
            min_matches_for_geometry=self.min_matches_for_geometry
        )

        pose_mask = geo['pose_mask']
        inlier_pts0 = pts0[pose_mask] if len(pose_mask) == len(pts0) else pts0
        inlier_pts1 = pts1[pose_mask] if len(pose_mask) == len(pts1) else pts1

        if len(match_scores) > 0 and len(pose_mask) == len(match_scores):
            valid_scores = match_scores[pose_mask]
            valid_scores = valid_scores[valid_scores >= 0.0]
            mean_match_score = float(valid_scores.mean()) if len(valid_scores) > 0 else float(match_result['mean_match_score'])
        else:
            mean_match_score = float(match_result['mean_match_score'])

        h0, w0 = prev_img.shape[:2]
        h1, w1 = curr_img.shape[:2]

        coverage0 = compute_coverage(inlier_pts0, w0, h0, rows=self.grid_rows, cols=self.grid_cols)
        coverage1 = compute_coverage(inlier_pts1, w1, h1, rows=self.grid_rows, cols=self.grid_cols)

        parallax_mean_px, parallax_median_px = compute_parallax_stats(inlier_pts0, inlier_pts1)

        R_odom_rel, t_odom_rel = compute_odom_relative_pose(prev_row, curr_row)

        odom_translation_m = float(translation_m) if translation_m is not None else -1.0
        odom_rotation_deg = float(rotation_deg) if rotation_deg is not None else -1.0

        if geo['pose_ok'] and R_odom_rel is not None and t_odom_rel is not None:
            e_rot_iv_deg = rotation_distance_deg(geo['R_vis'], R_odom_rel)
            e_trans_dir_iv_deg = vector_angle_deg(geo['t_vis_unit'], t_odom_rel)
        else:
            e_rot_iv_deg = -1.0
            e_trans_dir_iv_deg = -1.0

        cache_row = {
            'pair_id': int(kf_prev_id),
            'kf_prev_id': int(kf_prev_id),
            'kf_curr_id': int(kf_curr_id),
            'src_prev_id': int(prev_row.get('source_frame_id', -1)),
            'src_curr_id': int(curr_row.get('source_frame_id', -1)),
            'image_prev': prev_row.get('image_file', ''),
            'image_curr': curr_row.get('image_file', ''),
            'timestamp_prev': prev_row.get('timestamp_token', ''),
            'timestamp_curr': curr_row.get('timestamp_token', ''),

            'odom_translation_m': odom_translation_m,
            'odom_rotation_deg': odom_rotation_deg,

            'num_keypoints0': int(match_result['num_keypoints0']),
            'num_keypoints1': int(match_result['num_keypoints1']),
            'num_matches': int(match_result['num_matches']),
            'num_inliers': int(geo['num_inliers']),
            'match_inlier_ratio': float(geo['inlier_ratio']),
            'mean_match_score': float(mean_match_score),

            'coverage0': float(coverage0),
            'coverage1': float(coverage1),
            'parallax_mean_px': float(parallax_mean_px),
            'parallax_median_px': float(parallax_median_px),

            'geo_error_mean': float(geo['geo_error_mean']),
            'geo_error_median': float(geo['geo_error_median']),

            'vis_pose_ok': int(geo['pose_ok']),
            'vis_rot_deg': float(geo['vis_rot_deg']),
            'e_rot_iv_deg': float(e_rot_iv_deg),
            'e_trans_dir_iv_deg': float(e_trans_dir_iv_deg),
        }

        return cache_row

    def reset(self):
        self.last_kf_row = None
        self.last_kf_img = None
        self.last_kf_feats = None
        self.keyframe_id = 0
        self.K = None

        if self.cache_csv_path is not None:
            self._init_cache_file()

    def update(self, frame_packet):
        if frame_packet is None:
            raise ValueError('frame_packet 是 None')

        if 'image_bgr' not in frame_packet:
            raise ValueError("frame_packet 缺少 'image_bgr'")

        img = frame_packet['image_bgr']
        if img is None:
            raise ValueError("frame_packet['image_bgr'] 是 None")

        source_frame_id = frame_packet.get('source_frame_id', -1)

        if self.last_kf_row is None:
            curr_feats = self.matcher.extract(img)

            result = {
                'accepted': True,
                'reason': 'first_frame',
                'keyframe_id': self.keyframe_id,
                'source_frame_id': source_frame_id,
                'translation_m': 0.0,
                'rotation_deg': 0.0,
                'num_matches': -1,
                'inlier_ratio': -1.0,
                'match_result': None,
                'accepted_pair_cache': None,
                'cache_written': False,
                'cache_error': '',
            }

            self.last_kf_row = frame_packet
            self.last_kf_img = img
            self.last_kf_feats = curr_feats
            self.keyframe_id += 1
            return result

        translation_m = compute_translation(self.last_kf_row, frame_packet)
        rotation_deg = compute_rotation_deg(self.last_kf_row, frame_packet)

        has_motion_info = (translation_m is not None) or (rotation_deg is not None)

        too_small_translation = (translation_m is not None and translation_m < self.prefilter_translation_m)
        too_small_rotation = (rotation_deg is not None and rotation_deg < self.prefilter_rotation_deg)

        if has_motion_info:
            checks = []
            if translation_m is not None:
                checks.append(too_small_translation)
            if rotation_deg is not None:
                checks.append(too_small_rotation)

            if len(checks) > 0 and all(checks):
                return {
                    'accepted': False,
                    'reason': 'prefilter_skip',
                    'keyframe_id': None,
                    'source_frame_id': source_frame_id,
                    'translation_m': translation_m,
                    'rotation_deg': rotation_deg,
                    'num_matches': 0,
                    'inlier_ratio': 0.0,
                    'match_result': None,
                    'accepted_pair_cache': None,
                    'cache_written': False,
                    'cache_error': '',
                }

        curr_feats = self.matcher.extract(img)

        try:
            result = self.matcher.match_features(self.last_kf_feats, curr_feats)
        except Exception as e:
            return {
                'accepted': False,
                'reason': f'matcher_fail: {e}',
                'keyframe_id': None,
                'source_frame_id': source_frame_id,
                'translation_m': translation_m,
                'rotation_deg': rotation_deg,
                'num_matches': 0,
                'inlier_ratio': 0.0,
                'match_result': None,
                'accepted_pair_cache': None,
                'cache_written': False,
                'cache_error': '',
            }

        enough_visual = (
            result['num_matches'] >= self.min_matches and
            result['inlier_ratio'] >= self.min_inlier_ratio
        )

        if not has_motion_info:
            enough_motion = True
        else:
            motion_checks = []
            if translation_m is not None:
                motion_checks.append(translation_m >= self.min_translation_m)
            if rotation_deg is not None:
                motion_checks.append(rotation_deg >= self.min_rotation_deg)

            enough_motion = any(motion_checks) if len(motion_checks) > 0 else True

        if enough_visual and enough_motion:
            kf_prev_id = self.keyframe_id - 1
            kf_curr_id = self.keyframe_id

            accepted_pair_cache = None
            cache_written = False
            cache_error = ''

            try:
                accepted_pair_cache = self._build_accepted_pair_cache(
                    prev_row=self.last_kf_row,
                    curr_row=frame_packet,
                    prev_img=self.last_kf_img,
                    curr_img=img,
                    match_result=result,
                    translation_m=translation_m,
                    rotation_deg=rotation_deg,
                    kf_prev_id=kf_prev_id,
                    kf_curr_id=kf_curr_id,
                )
                self._append_cache_row(accepted_pair_cache)
                cache_written = self.cache_csv_path is not None
            except Exception as e:
                cache_error = str(e)

            decision = {
                'accepted': True,
                'reason': 'accept',
                'keyframe_id': self.keyframe_id,
                'source_frame_id': source_frame_id,
                'translation_m': translation_m,
                'rotation_deg': rotation_deg,
                'num_matches': result['num_matches'],
                'inlier_ratio': result['inlier_ratio'],
                'match_result': result,
                'accepted_pair_cache': accepted_pair_cache,
                'cache_written': cache_written,
                'cache_error': cache_error,
            }

            self.last_kf_row = frame_packet
            self.last_kf_img = img
            self.last_kf_feats = curr_feats
            self.keyframe_id += 1
            return decision

        if not enough_visual and not enough_motion:
            reason = 'reject_visual_and_motion'
        elif not enough_visual:
            reason = 'reject_visual'
        else:
            reason = 'reject_motion'

        return {
            'accepted': False,
            'reason': reason,
            'keyframe_id': None,
            'source_frame_id': source_frame_id,
            'translation_m': translation_m,
            'rotation_deg': rotation_deg,
            'num_matches': result['num_matches'],
            'inlier_ratio': result['inlier_ratio'],
            'match_result': result,
            'accepted_pair_cache': None,
            'cache_written': False,
            'cache_error': '',
        }