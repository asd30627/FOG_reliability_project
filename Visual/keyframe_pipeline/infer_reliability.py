import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from build_reliability_dataset import NONNEGATIVE_FEATURES
from train_reliability_model import build_model


class ReliabilityInferencer:
    def __init__(
        self,
        sequence_dir,
        feature_csv='',
        dataset_dir='',
        checkpoint='',
        out_dir='',
        tau=0.5,
        batch_size=256,
        device=None,
    ):
        self.sequence_dir = Path(sequence_dir).expanduser().resolve()
        self.feature_csv = Path(feature_csv).expanduser().resolve() if feature_csv else self.sequence_dir / 'features' / 'local_visual_features.csv'
        self.dataset_dir = Path(dataset_dir).expanduser().resolve() if dataset_dir else self.sequence_dir / 'reliability_dataset'
        self.checkpoint = Path(checkpoint).expanduser().resolve() if checkpoint else self.dataset_dir / 'model_runs' / 'gru' / 'best.pt'
        self.out_dir = Path(out_dir).expanduser().resolve() if out_dir else self.sequence_dir / 'reliability_inference'

        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.stats_json = self.dataset_dir / 'dataset_stats.json'
        self.debug_csv = self.dataset_dir / 'feature_debug_with_target.csv'

        self.out_dir.mkdir(parents=True, exist_ok=True)

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
    def is_valid_feature_value(name: str, value: float) -> bool:
        if not np.isfinite(value):
            return False
        if name in NONNEGATIVE_FEATURES and value < 0.0:
            return False
        return True

    def check_required_files(self):
        if not self.feature_csv.exists():
            raise FileNotFoundError(f'找不到 feature_csv: {self.feature_csv}')
        if not self.stats_json.exists():
            raise FileNotFoundError(f'找不到 dataset_stats.json: {self.stats_json}')
        if not self.checkpoint.exists():
            raise FileNotFoundError(f'找不到 checkpoint: {self.checkpoint}')

    def build_feature_table(self, rows, feature_names):
        rows_sorted = sorted(rows, key=lambda r: int(self.to_float(r.get('pair_id', 0), 0)))

        pair_ids = []
        meta = {}
        cols = {name: [] for name in feature_names}

        for row in rows_sorted:
            pid = int(self.to_float(row.get('pair_id', 0), 0))
            pair_ids.append(pid)

            meta[pid] = {
                'pair_id': pid,
                'kf_prev_id': int(self.to_float(row.get('kf_prev_id', -1), -1)),
                'kf_curr_id': int(self.to_float(row.get('kf_curr_id', -1), -1)),
                'src_prev_id': int(self.to_float(row.get('src_prev_id', -1), -1)),
                'src_curr_id': int(self.to_float(row.get('src_curr_id', -1), -1)),
                'image_prev': row.get('image_prev', ''),
                'image_curr': row.get('image_curr', ''),
                'timestamp_prev': row.get('timestamp_prev', ''),
                'timestamp_curr': row.get('timestamp_curr', ''),
            }

            for name in feature_names:
                if name.startswith('valid_'):
                    base_name = name[len('valid_'):]
                    v = self.to_float(row.get(base_name, np.nan), np.nan)
                    cols[name].append(1.0 if self.is_valid_feature_value(base_name, v) else 0.0)
                else:
                    v = self.to_float(row.get(name, np.nan), np.nan)
                    cols[name].append(v if self.is_valid_feature_value(name, v) else np.nan)

        table = {
            'pair_ids': np.array(pair_ids, dtype=np.int32),
            'meta': meta,
        }

        for name in feature_names:
            table[name] = np.array(cols[name], dtype=np.float64)

        return table

    @staticmethod
    def apply_fill_values(table, feature_names, fill_values):
        X_cols = []

        for name in feature_names:
            arr = table[name].copy()

            if name.startswith('valid_'):
                fill_v = 0.0
            else:
                fill_v = float(fill_values.get(name, 0.0))

            arr[~np.isfinite(arr)] = fill_v
            X_cols.append(arr.reshape(-1, 1))

        return np.concatenate(X_cols, axis=1).astype(np.float32)

    @staticmethod
    def build_sequences(X, pair_ids, seq_len):
        xs = []
        seq_meta = []

        for end_idx in range(seq_len - 1, len(X)):
            start_idx = end_idx - seq_len + 1
            xs.append(X[start_idx:end_idx + 1])

            seq_meta.append({
                'start_pair_id': int(pair_ids[start_idx]),
                'end_pair_id': int(pair_ids[end_idx]),
                'end_index': int(end_idx),
            })

        if len(xs) == 0:
            return np.empty((0, seq_len, X.shape[1]), dtype=np.float32), []

        return np.stack(xs).astype(np.float32), seq_meta

    @staticmethod
    def standardize_sequences(X_seq, mean, std):
        mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
        std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
        std = np.where(std < 1e-6, 1.0, std)
        return ((X_seq - mean) / std).astype(np.float32)

    def load_soft_target_map(self):
        if not self.debug_csv.exists():
            return {}

        rows = self.read_csv_rows(self.debug_csv)
        target_map = {}

        for row in rows:
            pid = int(self.to_float(row.get('pair_id', -1), -1))
            if pid < 0:
                continue
            target_map[pid] = float(self.to_float(row.get('soft_target', np.nan), np.nan))

        return target_map

    def infer_in_batches(self, model, X_seq):
        model.eval()
        preds = []

        with torch.no_grad():
            for start in range(0, len(X_seq), self.batch_size):
                end = min(start + self.batch_size, len(X_seq))
                xb = torch.from_numpy(X_seq[start:end]).float().to(self.device)
                pred = model(xb).detach().cpu().numpy()
                preds.append(pred)

        if len(preds) == 0:
            return np.array([], dtype=np.float32)

        preds = np.concatenate(preds, axis=0).astype(np.float32).reshape(-1)
        preds = np.clip(preds, 0.0, 1.0)
        return preds

    def load_stats_and_checkpoint(self):
        with open(self.stats_json, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        ckpt = torch.load(self.checkpoint, map_location=self.device)

        feature_names = ckpt.get('feature_names', stats['feature_names'])
        seq_len = int(ckpt['seq_len'])
        input_dim = int(ckpt['input_dim'])
        hidden_dim = int(ckpt['hidden_dim'])
        num_layers = int(ckpt['num_layers'])
        dropout = float(ckpt['dropout'])
        model_type = str(ckpt['model_type'])

        if len(feature_names) != input_dim:
            raise ValueError(
                f'checkpoint 內 input_dim={input_dim}，但 feature_names 數量={len(feature_names)}，不一致'
            )

        if len(stats['standardize_mean']) != len(feature_names):
            raise ValueError(
                f'standardize_mean 長度={len(stats["standardize_mean"])}, '
                f'feature_names 長度={len(feature_names)}，不一致'
            )

        if len(stats['standardize_std']) != len(feature_names):
            raise ValueError(
                f'standardize_std 長度={len(stats["standardize_std"])}, '
                f'feature_names 長度={len(feature_names)}，不一致'
            )

        return {
            'stats': stats,
            'ckpt': ckpt,
            'feature_names': feature_names,
            'seq_len': seq_len,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'model_type': model_type,
        }

    def build_model_from_checkpoint(self, meta):
        model = build_model(
            model_type=meta['model_type'],
            seq_len=meta['seq_len'],
            input_dim=meta['input_dim'],
            hidden_dim=meta['hidden_dim'],
            num_layers=meta['num_layers'],
            dropout=meta['dropout'],
        ).to(self.device)

        model.load_state_dict(meta['ckpt']['model_state_dict'])
        return model

    def save_prediction_csv(self, out_csv, seq_meta, table, preds):
        soft_target_map = self.load_soft_target_map()

        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sample_index',
                'start_pair_id',
                'end_pair_id',
                'kf_prev_id',
                'kf_curr_id',
                'src_prev_id',
                'src_curr_id',
                'image_prev',
                'image_curr',
                'timestamp_prev',
                'timestamp_curr',
                'w_pred',
                'gate_pass',
                'soft_target_proxy',
            ])

            for i, meta in enumerate(seq_meta):
                end_pid = meta['end_pair_id']
                pair_meta = table['meta'][end_pid]
                w_pred = float(preds[i])

                writer.writerow([
                    i,
                    meta['start_pair_id'],
                    end_pid,
                    pair_meta['kf_prev_id'],
                    pair_meta['kf_curr_id'],
                    pair_meta['src_prev_id'],
                    pair_meta['src_curr_id'],
                    pair_meta['image_prev'],
                    pair_meta['image_curr'],
                    pair_meta['timestamp_prev'],
                    pair_meta['timestamp_curr'],
                    w_pred,
                    int(w_pred >= self.tau),
                    soft_target_map.get(end_pid, np.nan),
                ])

    def save_summary(self, preds):
        summary = {
            'sequence_dir': str(self.sequence_dir),
            'feature_csv': str(self.feature_csv),
            'dataset_dir': str(self.dataset_dir),
            'checkpoint': str(self.checkpoint),
            'num_predictions': int(len(preds)),
            'tau': float(self.tau),
            'w_pred_mean': float(np.mean(preds)) if len(preds) > 0 else 0.0,
            'w_pred_std': float(np.std(preds)) if len(preds) > 0 else 0.0,
            'gate_pass_count': int(np.sum(preds >= self.tau)),
            'gate_pass_ratio': float(np.mean(preds >= self.tau)) if len(preds) > 0 else 0.0,
        }

        summary_path = self.out_dir / 'summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary_path

    def run(self):
        self.check_required_files()

        meta = self.load_stats_and_checkpoint()
        rows = self.read_csv_rows(self.feature_csv)

        table = self.build_feature_table(rows, meta['feature_names'])
        X_raw = self.apply_fill_values(table, meta['feature_names'], meta['stats']['fill_values'])

        X_seq, seq_meta = self.build_sequences(
            X_raw,
            table['pair_ids'],
            seq_len=meta['seq_len']
        )

        if len(X_seq) == 0:
            raise RuntimeError(
                f'建立 sequence 後為空，請確認 feature rows 數量與 seq_len={meta["seq_len"]}'
            )

        X_seq = self.standardize_sequences(
            X_seq,
            mean=meta['stats']['standardize_mean'],
            std=meta['stats']['standardize_std']
        )

        model = self.build_model_from_checkpoint(meta)
        preds = self.infer_in_batches(model, X_seq)

        out_csv = self.out_dir / 'reliability_predictions.csv'
        self.save_prediction_csv(out_csv, seq_meta, table, preds)
        summary_path = self.save_summary(preds)

        print(f'完成，reliability 推論輸出到: {out_csv}')
        print(f'summary: {summary_path}')

        return {
            'prediction_csv': out_csv,
            'summary_json': summary_path,
            'num_predictions': int(len(preds)),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_dir', type=str, required=True, help='例如: /mnt/sata4t/dataset/sequence_001')
    parser.add_argument('--feature_csv', type=str, default='', help='若留空，預設用 sequence_dir/features/local_visual_features.csv')
    parser.add_argument('--dataset_dir', type=str, default='', help='若留空，預設用 sequence_dir/reliability_dataset')
    parser.add_argument('--checkpoint', type=str, default='', help='若留空，預設用 dataset_dir/model_runs/gru/best.pt')
    parser.add_argument('--out_dir', type=str, default='', help='若留空，預設輸出到 sequence_dir/reliability_inference')
    parser.add_argument('--tau', type=float, default=0.5, help='hard gate 門檻')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def main():
    args = parse_args()

    inferencer = ReliabilityInferencer(
        sequence_dir=args.sequence_dir,
        feature_csv=args.feature_csv,
        dataset_dir=args.dataset_dir,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        tau=args.tau,
        batch_size=args.batch_size,
        device=args.device,
    )
    inferencer.run()


if __name__ == '__main__':
    main()