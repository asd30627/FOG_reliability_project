import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from train_reliability_model import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sequence_dir',
        type=str,
        required=True,
        help='例如: /mnt/sata4t/dataset/sequence_001'
    )
    parser.add_argument(
        '--feature_csv',
        type=str,
        default='',
        help='若留空，預設用 sequence_dir/features/local_visual_features.csv'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='',
        help='若留空，預設用 sequence_dir/reliability_dataset'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='',
        help='若留空，預設用 dataset_dir/model_runs/gru/best.pt'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='',
        help='若留空，預設輸出到 sequence_dir/reliability_inference'
    )
    parser.add_argument('--tau', type=float, default=0.5, help='hard gate 門檻')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def read_csv_rows(path: Path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def build_feature_table(rows, feature_names):
    rows_sorted = sorted(rows, key=lambda r: int(to_float(r.get('pair_id', 0), 0)))

    pair_ids = []
    meta = {}
    cols = {name: [] for name in feature_names}

    for row in rows_sorted:
        pid = int(to_float(row.get('pair_id', 0), 0))
        pair_ids.append(pid)

        meta[pid] = {
            'pair_id': pid,
            'kf_prev_id': int(to_float(row.get('kf_prev_id', -1), -1)),
            'kf_curr_id': int(to_float(row.get('kf_curr_id', -1), -1)),
            'src_prev_id': int(to_float(row.get('src_prev_id', -1), -1)),
            'src_curr_id': int(to_float(row.get('src_curr_id', -1), -1)),
            'image_prev': row.get('image_prev', ''),
            'image_curr': row.get('image_curr', ''),
            'timestamp_prev': row.get('timestamp_prev', ''),
            'timestamp_curr': row.get('timestamp_curr', ''),
        }

        for name in feature_names:
            v = to_float(row.get(name, np.nan), np.nan)
            if np.isfinite(v) and v < 0.0:
                # 這些欄位在前面流程裡用 -1 表示 invalid，這裡轉成 NaN
                v = np.nan
            cols[name].append(v)

    table = {
        'pair_ids': np.array(pair_ids, dtype=np.int32),
        'meta': meta,
    }
    for name in feature_names:
        table[name] = np.array(cols[name], dtype=np.float64)

    return table


def apply_fill_values(table, feature_names, fill_values):
    X_cols = []
    for name in feature_names:
        arr = table[name].copy()
        fill_v = float(fill_values.get(name, 0.0))
        bad = ~np.isfinite(arr)
        arr[bad] = fill_v
        X_cols.append(arr.reshape(-1, 1))
    X = np.concatenate(X_cols, axis=1).astype(np.float32)
    return X


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


def standardize_sequences(X_seq, mean, std):
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
    std = np.where(std < 1e-6, 1.0, std)
    return ((X_seq - mean) / std).astype(np.float32)


def load_soft_target_map(debug_csv: Path):
    if not debug_csv.exists():
        return {}

    rows = read_csv_rows(debug_csv)
    target_map = {}
    for row in rows:
        pid = int(to_float(row.get('pair_id', -1), -1))
        if pid < 0:
            continue
        target_map[pid] = float(to_float(row.get('soft_target', np.nan), np.nan))
    return target_map


def infer_in_batches(model, X_seq, batch_size, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for start in range(0, len(X_seq), batch_size):
            end = min(start + batch_size, len(X_seq))
            xb = torch.from_numpy(X_seq[start:end]).float().to(device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)

    if len(preds) == 0:
        return np.array([], dtype=np.float32)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = np.clip(preds, 0.0, 1.0)
    return preds


def main():
    args = parse_args()

    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    feature_csv = Path(args.feature_csv).expanduser().resolve() if args.feature_csv else sequence_dir / 'features' / 'local_visual_features.csv'
    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else sequence_dir / 'reliability_dataset'
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else dataset_dir / 'model_runs' / 'gru' / 'best.pt'
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else sequence_dir / 'reliability_inference'
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_json = dataset_dir / 'dataset_stats.json'
    debug_csv = dataset_dir / 'feature_debug_with_target.csv'

    if not feature_csv.exists():
        raise FileNotFoundError(f'找不到 feature_csv: {feature_csv}')
    if not stats_json.exists():
        raise FileNotFoundError(f'找不到 dataset_stats.json: {stats_json}')
    if not checkpoint.exists():
        raise FileNotFoundError(f'找不到 checkpoint: {checkpoint}')

    with open(stats_json, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    ckpt = torch.load(checkpoint, map_location=args.device)

    feature_names = ckpt.get('feature_names', stats['feature_names'])
    seq_len = int(ckpt['seq_len'])
    input_dim = int(ckpt['input_dim'])
    hidden_dim = int(ckpt['hidden_dim'])
    num_layers = int(ckpt['num_layers'])
    dropout = float(ckpt['dropout'])
    model_type = str(ckpt['model_type'])

    rows = read_csv_rows(feature_csv)
    table = build_feature_table(rows, feature_names)
    X_raw = apply_fill_values(table, feature_names, stats['fill_values'])
    X_seq, seq_meta = build_sequences(X_raw, table['pair_ids'], seq_len=seq_len)

    if len(X_seq) == 0:
        raise RuntimeError(f'建立 sequence 後為空，請確認 feature rows 數量與 seq_len={seq_len}')

    X_seq = standardize_sequences(
        X_seq,
        mean=stats['standardize_mean'],
        std=stats['standardize_std']
    )

    model = build_model(
        model_type=model_type,
        seq_len=seq_len,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(args.device)
    model.load_state_dict(ckpt['model_state_dict'])

    preds = infer_in_batches(model, X_seq, batch_size=args.batch_size, device=args.device)

    soft_target_map = load_soft_target_map(debug_csv)

    out_csv = out_dir / 'reliability_predictions.csv'
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
            'soft_target_proxy'
        ])

        for i, meta in enumerate(seq_meta):
            end_pid = meta['end_pair_id']
            pair_meta = table['meta'][end_pid]
            w_pred = float(preds[i])
            gate_pass = int(w_pred >= args.tau)
            soft_target_proxy = soft_target_map.get(end_pid, np.nan)

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
                gate_pass,
                soft_target_proxy,
            ])

    summary = {
        'sequence_dir': str(sequence_dir),
        'feature_csv': str(feature_csv),
        'dataset_dir': str(dataset_dir),
        'checkpoint': str(checkpoint),
        'model_type': model_type,
        'seq_len': seq_len,
        'num_predictions': int(len(preds)),
        'tau': float(args.tau),
        'w_pred_mean': float(np.mean(preds)) if len(preds) > 0 else 0.0,
        'w_pred_std': float(np.std(preds)) if len(preds) > 0 else 0.0,
        'gate_pass_count': int(np.sum(preds >= args.tau)),
        'gate_pass_ratio': float(np.mean(preds >= args.tau)) if len(preds) > 0 else 0.0,
    }

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f'完成，reliability 推論輸出到: {out_csv}')
    print(f'summary: {out_dir / "summary.json"}')


if __name__ == '__main__':
    main()