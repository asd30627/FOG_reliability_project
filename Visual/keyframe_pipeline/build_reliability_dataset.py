import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


DEFAULT_BASE_FEATURE_NAMES = [
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

NONNEGATIVE_FEATURES = {
    'odom_translation_m', 'odom_rotation_deg', 'num_keypoints0', 'num_keypoints1',
    'num_matches', 'num_inliers', 'match_inlier_ratio', 'mean_match_score',
    'coverage0', 'coverage1', 'parallax_mean_px', 'parallax_median_px',
    'geo_error_mean', 'geo_error_median', 'vis_pose_ok', 'vis_rot_deg',
    'e_rot_iv_deg', 'e_trans_dir_iv_deg', 'blur0', 'blur1', 'brightness0',
    'brightness1', 'texture0', 'texture1', 'edge_density0', 'edge_density1',
}


class ReliabilityDatasetBuilder:
    def __init__(
        self,
        sequence_dir,
        feature_csv='',
        out_dir='',
        seq_len=8,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        min_rows=16,
        use_validity_mask=False,
        base_feature_names=None,
    ):
        self.sequence_dir = Path(sequence_dir).expanduser().resolve()
        self.feature_csv = Path(feature_csv).expanduser().resolve() if feature_csv else self.sequence_dir / 'features' / 'local_visual_features.csv'
        self.out_dir = Path(out_dir).expanduser().resolve() if out_dir else self.sequence_dir / 'reliability_dataset'

        self.seq_len = int(seq_len)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.min_rows = int(min_rows)
        self.use_validity_mask = bool(use_validity_mask)

        self.base_feature_names = list(base_feature_names) if base_feature_names is not None else list(DEFAULT_BASE_FEATURE_NAMES)

    @staticmethod
    def read_csv_rows(path: Path):
        with open(path, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))

    @staticmethod
    def to_float(value, default=np.nan):
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def robust_positive_scale(values, default_value):
        values = np.asarray(values, dtype=np.float64)
        valid = values[np.isfinite(values) & (values >= 0.0)]
        if len(valid) == 0:
            return float(default_value)
        q75 = np.percentile(valid, 75)
        q90 = np.percentile(valid, 90)
        return max(float(q75), float(q90) * 0.5, float(default_value))

    @staticmethod
    def exp_good_from_error(values, scale, invalid_fill=0.0):
        values = np.asarray(values, dtype=np.float64)
        out = np.full(values.shape, invalid_fill, dtype=np.float64)
        valid = np.isfinite(values) & (values >= 0.0)
        out[valid] = np.exp(-values[valid] / max(scale, 1e-6))
        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def clip01(values):
        return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)

    def compute_proxy_soft_target(self, table: Dict[str, np.ndarray]):
        vis_pose_ok = self.clip01(table['vis_pose_ok'])
        inlier = self.clip01(table['match_inlier_ratio'])

        mean_match_score = np.asarray(table['mean_match_score'], dtype=np.float64)
        mean_match_score = np.where(mean_match_score < 0.0, 0.0, mean_match_score)
        mean_match_score = self.clip01(mean_match_score)

        coverage = 0.5 * (self.clip01(table['coverage0']) + self.clip01(table['coverage1']))

        parallax_scale = self.robust_positive_scale(table['parallax_mean_px'], default_value=10.0)
        parallax_good = 1.0 - np.exp(-np.maximum(table['parallax_mean_px'], 0.0) / parallax_scale)
        parallax_good = self.clip01(parallax_good)

        geo_scale = self.robust_positive_scale(table['geo_error_mean'], default_value=1.0)
        geo_good = self.exp_good_from_error(table['geo_error_mean'], scale=geo_scale, invalid_fill=0.0)

        rot_scale = self.robust_positive_scale(table['e_rot_iv_deg'], default_value=10.0)
        rot_good = self.exp_good_from_error(table['e_rot_iv_deg'], scale=rot_scale, invalid_fill=0.0)

        tdir_scale = self.robust_positive_scale(table['e_trans_dir_iv_deg'], default_value=20.0)
        tdir_good = self.exp_good_from_error(table['e_trans_dir_iv_deg'], scale=tdir_scale, invalid_fill=0.0)

        weights = {
            'pose': 0.15,
            'inlier': 0.20,
            'score': 0.10,
            'coverage': 0.10,
            'parallax': 0.10,
            'geo': 0.10,
            'rot': 0.15,
            'tdir': 0.10,
        }
        total_w = sum(weights.values())

        soft_target = (
            weights['pose'] * vis_pose_ok +
            weights['inlier'] * inlier +
            weights['score'] * mean_match_score +
            weights['coverage'] * coverage +
            weights['parallax'] * parallax_good +
            weights['geo'] * geo_good +
            weights['rot'] * rot_good +
            weights['tdir'] * tdir_good
        ) / total_w
        soft_target = self.clip01(soft_target)

        scales = {
            'parallax_scale': parallax_scale,
            'geo_scale': geo_scale,
            'rot_scale': rot_scale,
            'tdir_scale': tdir_scale,
        }

        debug_components = {
            'target_pose_ok': vis_pose_ok,
            'target_inlier': inlier,
            'target_match_score': mean_match_score,
            'target_coverage': coverage,
            'target_parallax_good': parallax_good,
            'target_geo_good': geo_good,
            'target_rot_good': rot_good,
            'target_tdir_good': tdir_good,
            'soft_target': soft_target,
        }
        return soft_target, scales, debug_components

    @staticmethod
    def _is_valid_feature_value(name: str, value: float) -> bool:
        if not np.isfinite(value):
            return False
        if name in NONNEGATIVE_FEATURES and value < 0.0:
            return False
        return True

    def rows_to_table(self, rows: List[Dict[str, str]], feature_names: List[str]):
        table: Dict[str, np.ndarray] = {}

        pair_ids = []
        for idx, row in enumerate(rows):
            pair_ids.append(int(self.to_float(row.get('pair_id', idx), idx)))
        table['pair_id'] = np.array(pair_ids, dtype=np.int32)

        for name in feature_names:
            values = []
            validity = []
            for row in rows:
                v = self.to_float(row.get(name, np.nan), np.nan)
                valid = self._is_valid_feature_value(name, v)
                validity.append(1.0 if valid else 0.0)
                values.append(v if valid else np.nan)

            table[name] = np.array(values, dtype=np.float64)
            table[f'valid_{name}'] = np.array(validity, dtype=np.float64)

        return table

    def fill_nan_with_column_median(self, table: Dict[str, np.ndarray], feature_names: List[str]):
        fill_values = {}
        for name in feature_names:
            arr = table[name].copy()
            valid = arr[np.isfinite(arr)]
            median = float(np.median(valid)) if len(valid) > 0 else 0.0
            fill_values[name] = median
            arr[~np.isfinite(arr)] = median
            table[name] = arr
        return fill_values

    def table_to_feature_matrix(self, table: Dict[str, np.ndarray], feature_names: List[str]):
        final_names = list(feature_names)
        mats = [table[name].reshape(-1, 1) for name in feature_names]

        if self.use_validity_mask:
            validity_names = [f'valid_{name}' for name in feature_names]
            final_names.extend(validity_names)
            mats.extend(table[name].reshape(-1, 1) for name in validity_names)

        X = np.concatenate(mats, axis=1).astype(np.float32)
        return X, final_names

    @staticmethod
    def build_sequence_samples(X, y, pair_ids, seq_len):
        xs, ys, end_pair_ids = [], [], []

        for end_idx in range(seq_len - 1, len(X)):
            start_idx = end_idx - seq_len + 1
            xs.append(X[start_idx:end_idx + 1])
            ys.append(y[end_idx])
            end_pair_ids.append(pair_ids[end_idx])

        if len(xs) == 0:
            return (
                np.empty((0, seq_len, X.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        return (
            np.stack(xs).astype(np.float32),
            np.array(ys, dtype=np.float32),
            np.array(end_pair_ids, dtype=np.int32),
        )

    @staticmethod
    def compute_standardization_stats(X_train):
        flat = X_train.reshape(-1, X_train.shape[-1]).astype(np.float64)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std[std < 1e-6] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    @staticmethod
    def apply_standardization(X, mean, std):
        return ((X - mean[None, None, :]) / std[None, None, :]).astype(np.float32)

    @staticmethod
    def save_split_npz(path, X, y, pair_ids, feature_names, seq_len):
        np.savez_compressed(
            path,
            X=X.astype(np.float32),
            y=y.astype(np.float32),
            pair_ids=pair_ids.astype(np.int32),
            feature_names=np.array(feature_names, dtype=object),
            seq_len=np.array([seq_len], dtype=np.int32),
        )

    def build(self):
        if not self.feature_csv.exists():
            raise FileNotFoundError(f'找不到 feature CSV: {self.feature_csv}')

        self.out_dir.mkdir(parents=True, exist_ok=True)

        rows = self.read_csv_rows(self.feature_csv)
        if len(rows) < self.min_rows:
            raise RuntimeError(f'feature 列數太少 ({len(rows)})，至少需要 {self.min_rows} 列')

        table = self.rows_to_table(rows, self.base_feature_names)

        soft_target, scales, debug_components = self.compute_proxy_soft_target(table)
        table['soft_target'] = soft_target

        fill_values = self.fill_nan_with_column_median(table, self.base_feature_names)
        X_raw, final_feature_names = self.table_to_feature_matrix(table, self.base_feature_names)
        y_raw = table['soft_target'].astype(np.float32)
        pair_ids = table['pair_id'].astype(np.int32)

        X_seq, y_seq, pid_seq = self.build_sequence_samples(X_raw, y_raw, pair_ids, seq_len=self.seq_len)
        if len(X_seq) == 0:
            raise RuntimeError('建立 sequence samples 後為空，請縮小 seq_len 或增加資料量')

        n = len(X_seq)
        if abs((self.train_ratio + self.val_ratio + self.test_ratio) - 1.0) > 1e-6:
            raise ValueError('train_ratio + val_ratio + test_ratio 必須等於 1')

        n_train = max(1, int(n * self.train_ratio))
        n_val = max(1, int(n * self.val_ratio))
        n_test = n - n_train - n_val

        if n_test <= 0:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            elif n_train > 1:
                n_train -= 1
            else:
                raise RuntimeError('資料太少，無法切 train/val/test')

        train_slice = slice(0, n_train)
        val_slice = slice(n_train, n_train + n_val)
        test_slice = slice(n_train + n_val, n)

        X_train, y_train, pid_train = X_seq[train_slice], y_seq[train_slice], pid_seq[train_slice]
        X_val, y_val, pid_val = X_seq[val_slice], y_seq[val_slice], pid_seq[val_slice]
        X_test, y_test, pid_test = X_seq[test_slice], y_seq[test_slice], pid_seq[test_slice]

        mean, std = self.compute_standardization_stats(X_train)
        X_train_std = self.apply_standardization(X_train, mean, std)
        X_val_std = self.apply_standardization(X_val, mean, std)
        X_test_std = self.apply_standardization(X_test, mean, std)

        self.save_split_npz(self.out_dir / 'train.npz', X_train_std, y_train, pid_train, final_feature_names, self.seq_len)
        self.save_split_npz(self.out_dir / 'val.npz', X_val_std, y_val, pid_val, final_feature_names, self.seq_len)
        self.save_split_npz(self.out_dir / 'test.npz', X_test_std, y_test, pid_test, final_feature_names, self.seq_len)

        stats = {
            'sequence_dir': str(self.sequence_dir),
            'feature_csv': str(self.feature_csv),
            'num_feature_rows': int(len(rows)),
            'num_sequence_samples': int(n),
            'seq_len': int(self.seq_len),
            'base_feature_names': self.base_feature_names,
            'feature_names': final_feature_names,
            'use_validity_mask': bool(self.use_validity_mask),
            'fill_values': fill_values,
            'standardize_mean': mean.tolist(),
            'standardize_std': std.tolist(),
            'soft_target_scales': scales,
            'split_sizes': {
                'train': int(len(X_train_std)),
                'val': int(len(X_val_std)),
                'test': int(len(X_test_std)),
            },
        }
        with open(self.out_dir / 'dataset_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        debug_header = ['pair_id'] + self.base_feature_names + [f'valid_{n}' for n in self.base_feature_names] + list(debug_components.keys())
        debug_csv = self.out_dir / 'feature_debug_with_target.csv'
        with open(debug_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(debug_header)
            for i in range(len(rows)):
                row_values = [int(pair_ids[i])]
                row_values += [float(table[name][i]) for name in self.base_feature_names]
                row_values += [float(table[f'valid_{name}'][i]) for name in self.base_feature_names]
                row_values += [float(debug_components[name][i]) for name in debug_components.keys()]
                writer.writerow(row_values)

        index_csv = self.out_dir / 'sequence_index.csv'
        with open(index_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['split', 'sample_index', 'end_pair_id', 'target'])
            for i, (pid, tgt) in enumerate(zip(pid_train, y_train)):
                writer.writerow(['train', i, int(pid), float(tgt)])
            for i, (pid, tgt) in enumerate(zip(pid_val, y_val)):
                writer.writerow(['val', i, int(pid), float(tgt)])
            for i, (pid, tgt) in enumerate(zip(pid_test, y_test)):
                writer.writerow(['test', i, int(pid), float(tgt)])

        print(f'完成，dataset 輸出到: {self.out_dir}')
        print('train/val/test npz 已建立')
        print(f'debug CSV: {debug_csv}')
        print(f'stats JSON: {self.out_dir / "dataset_stats.json"}')

        return {
            'rows': rows,
            'table': table,
            'stats': stats,
            'debug_components': debug_components,
        }