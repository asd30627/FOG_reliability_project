import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


FEATURE_HEADER = [
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

    'blur0',
    'blur1',
    'brightness0',
    'brightness1',
    'texture0',
    'texture1',
    'edge_density0',
    'edge_density1',
]


CACHED_FIELDS = [
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


class ReliabilityFeatureExtractor:
    def __init__(
        self,
        sequence_dir,
        cache_csv_path='',
        out_dir='',
        clean_output_dir=False,
    ):
        self.sequence_dir = Path(sequence_dir).expanduser().resolve()
        self.keyframes_dir = self.sequence_dir / 'keyframes'
        self.keyframe_images_dir = self.keyframes_dir / 'images'

        self.cache_csv_path = Path(cache_csv_path).expanduser().resolve() if cache_csv_path else self.keyframes_dir / 'accepted_pairs_cache.csv'

        self.feature_dir = Path(out_dir).expanduser().resolve() if out_dir else self.sequence_dir / 'features'
        self.feature_csv = self.feature_dir / 'local_visual_features.csv'

        self.clean_output_dir = bool(clean_output_dir)

    @staticmethod
    def ensure_dir(path: Path, clean=False):
        if clean and path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read_csv_rows(path: Path):
        with open(path, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))

    @staticmethod
    def to_float(value, default=-1.0):
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def to_int(value, default=-1):
        try:
            return int(float(value))
        except Exception:
            return default

    @staticmethod
    def compute_blur_score(img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def compute_brightness(img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())

    @staticmethod
    def compute_texture_score(img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        return float(mag.mean())

    @staticmethod
    def compute_edge_density(img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return float((edges > 0).mean())

    def _check_inputs(self):
        if not self.cache_csv_path.exists():
            raise FileNotFoundError(f'找不到 accepted_pairs_cache.csv: {self.cache_csv_path}')
        if not self.keyframe_images_dir.exists():
            raise FileNotFoundError(f'找不到 keyframe images 目錄: {self.keyframe_images_dir}')

    def _build_output_row(self, cache_row, img0, img1):
        blur0 = self.compute_blur_score(img0)
        blur1 = self.compute_blur_score(img1)
        brightness0 = self.compute_brightness(img0)
        brightness1 = self.compute_brightness(img1)
        texture0 = self.compute_texture_score(img0)
        texture1 = self.compute_texture_score(img1)
        edge_density0 = self.compute_edge_density(img0)
        edge_density1 = self.compute_edge_density(img1)

        row = [
            self.to_int(cache_row.get('pair_id', -1)),
            self.to_int(cache_row.get('kf_prev_id', -1)),
            self.to_int(cache_row.get('kf_curr_id', -1)),
            self.to_int(cache_row.get('src_prev_id', -1)),
            self.to_int(cache_row.get('src_curr_id', -1)),
            cache_row.get('image_prev', ''),
            cache_row.get('image_curr', ''),
            cache_row.get('timestamp_prev', ''),
            cache_row.get('timestamp_curr', ''),

            self.to_float(cache_row.get('odom_translation_m', -1.0)),
            self.to_float(cache_row.get('odom_rotation_deg', -1.0)),

            self.to_int(cache_row.get('num_keypoints0', -1)),
            self.to_int(cache_row.get('num_keypoints1', -1)),
            self.to_int(cache_row.get('num_matches', -1)),
            self.to_int(cache_row.get('num_inliers', -1)),
            self.to_float(cache_row.get('match_inlier_ratio', -1.0)),
            self.to_float(cache_row.get('mean_match_score', -1.0)),

            self.to_float(cache_row.get('coverage0', -1.0)),
            self.to_float(cache_row.get('coverage1', -1.0)),
            self.to_float(cache_row.get('parallax_mean_px', -1.0)),
            self.to_float(cache_row.get('parallax_median_px', -1.0)),

            self.to_float(cache_row.get('geo_error_mean', -1.0)),
            self.to_float(cache_row.get('geo_error_median', -1.0)),

            self.to_int(cache_row.get('vis_pose_ok', 0)),
            self.to_float(cache_row.get('vis_rot_deg', -1.0)),
            self.to_float(cache_row.get('e_rot_iv_deg', -1.0)),
            self.to_float(cache_row.get('e_trans_dir_iv_deg', -1.0)),

            blur0,
            blur1,
            brightness0,
            brightness1,
            texture0,
            texture1,
            edge_density0,
            edge_density1,
        ]
        return row

    def run(self):
        self._check_inputs()
        self.ensure_dir(self.feature_dir, clean=self.clean_output_dir)

        cache_rows = self.read_csv_rows(self.cache_csv_path)
        if len(cache_rows) == 0:
            raise RuntimeError(f'accepted_pairs_cache.csv 是空的: {self.cache_csv_path}')

        with open(self.feature_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(FEATURE_HEADER)

            for cache_row in cache_rows:
                for field in CACHED_FIELDS:
                    if field not in cache_row:
                        raise KeyError(f'cache row 缺少必要欄位: {field}')

                img0_path = self.keyframe_images_dir / cache_row['image_prev']
                img1_path = self.keyframe_images_dir / cache_row['image_curr']

                if not img0_path.exists() or not img1_path.exists():
                    print(f'[WARN] 缺少影像，略過 pair {cache_row.get("pair_id", "?")}: {img0_path.name}, {img1_path.name}')
                    continue

                img0 = cv2.imread(str(img0_path))
                img1 = cv2.imread(str(img1_path))
                if img0 is None or img1 is None:
                    print(f'[WARN] 影像讀取失敗，略過 pair {cache_row.get("pair_id", "?")}: {img0_path.name}, {img1_path.name}')
                    continue

                out_row = self._build_output_row(cache_row, img0, img1)
                writer.writerow(out_row)

        print(f'完成，feature CSV 輸出到: {self.feature_csv}')

        return {
            'feature_csv': self.feature_csv,
            'num_pairs': len(cache_rows),
        }