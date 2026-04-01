import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    return {
        'X': data['X'].astype(np.float32),
        'y': data['y'].astype(np.float32),
        'pair_ids': data['pair_ids'].astype(np.int32),
        'feature_names': [str(x) for x in data['feature_names'].tolist()],
        'seq_len': int(data['seq_len'][0]),
    }


class SequenceDataset(Dataset):
    def __init__(self, X, y, pair_ids):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.pair_ids = torch.from_numpy(pair_ids).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.pair_ids[idx]


class MLPRegressor(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        flat_dim = seq_len * input_dim
        mid_dim = max(hidden_dim // 2, 16)

        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.net(x).squeeze(-1)


class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.1):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        _, h = self.gru(x)
        last_h = h[-1]
        return self.head(last_h).squeeze(-1)


class TemporalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


class TCNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.block1 = TemporalConvBlock(input_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout)
        self.block2 = TemporalConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)


def build_model(model_type, seq_len, input_dim, hidden_dim, num_layers, dropout):
    if model_type == 'mlp':
        return MLPRegressor(seq_len=seq_len, input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if model_type == 'gru':
        return GRURegressor(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if model_type == 'tcn':
        return TCNRegressor(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f'未知 model_type: {model_type}')


def mse_np(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def corrcoef_np(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


class ReliabilityModelTrainer:
    def __init__(
        self,
        dataset_dir,
        out_dir='',
        model_type='gru',
        epochs=80,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dim=64,
        num_layers=1,
        dropout=0.10,
        seed=42,
        device=None,
    ):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.model_type = model_type
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.seed = int(seed)
        self.device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.out_dir = Path(out_dir).expanduser().resolve() if out_dir else self.dataset_dir / 'model_runs' / self.model_type
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.best_path = self.out_dir / 'best.pt'

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.seq_len = None
        self.input_dim = None
        self.feature_names = None

        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

        self.history = []
        self.best_val_mse = float('inf')
        self.best_epoch = -1

    def load_data(self):
        train_path = self.dataset_dir / 'train.npz'
        val_path = self.dataset_dir / 'val.npz'
        test_path = self.dataset_dir / 'test.npz'

        if not train_path.exists():
            raise FileNotFoundError(f'找不到 {train_path}')
        if not val_path.exists():
            raise FileNotFoundError(f'找不到 {val_path}')
        if not test_path.exists():
            raise FileNotFoundError(f'找不到 {test_path}')

        self.train_data = load_split_npz(train_path)
        self.val_data = load_split_npz(val_path)
        self.test_data = load_split_npz(test_path)

        self.seq_len = self.train_data['seq_len']
        self.input_dim = self.train_data['X'].shape[-1]
        self.feature_names = self.train_data['feature_names']

        if self.val_data['seq_len'] != self.seq_len or self.test_data['seq_len'] != self.seq_len:
            raise ValueError('train/val/test 的 seq_len 不一致')

        if self.val_data['X'].shape[-1] != self.input_dim or self.test_data['X'].shape[-1] != self.input_dim:
            raise ValueError('train/val/test 的 input_dim 不一致')

        train_ds = SequenceDataset(self.train_data['X'], self.train_data['y'], self.train_data['pair_ids'])
        val_ds = SequenceDataset(self.val_data['X'], self.val_data['y'], self.val_data['pair_ids'])
        test_ds = SequenceDataset(self.test_data['X'], self.test_data['y'], self.test_data['pair_ids'])

        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def build(self):
        self.model = build_model(
            model_type=self.model_type,
            seq_len=self.seq_len,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def run_one_epoch(self, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        preds_all = []
        targets_all = []
        pair_ids_all = []

        for X, y, pair_ids in loader:
            X = X.to(self.device)
            y = y.to(self.device)

            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                pred = self.model(X)
                loss = self.criterion(pred, y)

                if train:
                    loss.backward()
                    self.optimizer.step()

            total_loss += float(loss.item()) * X.shape[0]
            preds_all.append(pred.detach().cpu().numpy())
            targets_all.append(y.detach().cpu().numpy())
            pair_ids_all.append(pair_ids.detach().cpu().numpy())

        preds_all = np.concatenate(preds_all, axis=0) if preds_all else np.array([], dtype=np.float32)
        targets_all = np.concatenate(targets_all, axis=0) if targets_all else np.array([], dtype=np.float32)
        pair_ids_all = np.concatenate(pair_ids_all, axis=0) if pair_ids_all else np.array([], dtype=np.int32)

        avg_loss = total_loss / max(len(loader.dataset), 1)

        metrics = {
            'loss': avg_loss,
            'mse': mse_np(targets_all, preds_all) if len(preds_all) > 0 else 0.0,
            'mae': mae_np(targets_all, preds_all) if len(preds_all) > 0 else 0.0,
            'corr': corrcoef_np(targets_all, preds_all) if len(preds_all) > 0 else 0.0,
        }

        return metrics, preds_all, targets_all, pair_ids_all

    @staticmethod
    def save_predictions_csv(path, pair_ids, targets, preds):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['pair_id', 'target', 'pred'])
            for pid, tgt, pred in zip(pair_ids, targets, preds):
                writer.writerow([int(pid), float(tgt), float(pred)])

    def train(self):
        for epoch in range(1, self.epochs + 1):
            train_metrics, _, _, _ = self.run_one_epoch(self.train_loader, train=True)
            val_metrics, _, _, _ = self.run_one_epoch(self.val_loader, train=False)

            row = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_mse': train_metrics['mse'],
                'train_mae': train_metrics['mae'],
                'train_corr': train_metrics['corr'],
                'val_loss': val_metrics['loss'],
                'val_mse': val_metrics['mse'],
                'val_mae': val_metrics['mae'],
                'val_corr': val_metrics['corr'],
            }
            self.history.append(row)

            print(
                f'[Epoch {epoch:03d}] '
                f'train_mse={train_metrics["mse"]:.6f} '
                f'val_mse={val_metrics["mse"]:.6f} '
                f'val_mae={val_metrics["mae"]:.6f} '
                f'val_corr={val_metrics["corr"]:.4f}'
            )

            if val_metrics['mse'] < self.best_val_mse:
                self.best_val_mse = val_metrics['mse']
                self.best_epoch = epoch

                torch.save(
                    {
                        'model_state_dict': self.model.state_dict(),
                        'model_type': self.model_type,
                        'seq_len': self.seq_len,
                        'input_dim': self.input_dim,
                        'hidden_dim': self.hidden_dim,
                        'num_layers': self.num_layers,
                        'dropout': self.dropout,
                        'feature_names': self.feature_names,
                        'best_epoch': self.best_epoch,
                        'best_val_mse': self.best_val_mse,
                    },
                    self.best_path
                )

    def save_history(self):
        if len(self.history) == 0:
            return None

        history_csv = self.out_dir / 'history.csv'
        with open(history_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)

        return history_csv

    def load_best_model(self):
        if not self.best_path.exists():
            raise FileNotFoundError(f'找不到最佳模型: {self.best_path}')

        ckpt = torch.load(self.best_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        return ckpt

    def evaluate_and_save_predictions(self):
        train_metrics, train_pred, train_tgt, train_pid = self.run_one_epoch(self.train_loader, train=False)
        val_metrics, val_pred, val_tgt, val_pid = self.run_one_epoch(self.val_loader, train=False)
        test_metrics, test_pred, test_tgt, test_pid = self.run_one_epoch(self.test_loader, train=False)

        self.save_predictions_csv(self.out_dir / 'train_predictions.csv', train_pid, train_tgt, train_pred)
        self.save_predictions_csv(self.out_dir / 'val_predictions.csv', val_pid, val_tgt, val_pred)
        self.save_predictions_csv(self.out_dir / 'test_predictions.csv', test_pid, test_tgt, test_pred)

        return {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
        }

    def save_summary(self, final_metrics):
        summary = {
            'dataset_dir': str(self.dataset_dir),
            'out_dir': str(self.out_dir),
            'model_type': self.model_type,
            'seq_len': int(self.seq_len),
            'input_dim': int(self.input_dim),
            'feature_names': self.feature_names,
            'epochs': int(self.epochs),
            'batch_size': int(self.batch_size),
            'lr': float(self.lr),
            'weight_decay': float(self.weight_decay),
            'hidden_dim': int(self.hidden_dim),
            'num_layers': int(self.num_layers),
            'dropout': float(self.dropout),
            'seed': int(self.seed),
            'device': str(self.device),
            'best_epoch': int(self.best_epoch),
            'best_val_mse': float(self.best_val_mse),
            'final_metrics': final_metrics,
        }

        summary_path = self.out_dir / 'summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary_path

    def run(self):
        set_seed(self.seed)

        self.load_data()
        self.build()
        self.train()
        history_csv = self.save_history()
        self.load_best_model()
        final_metrics = self.evaluate_and_save_predictions()
        summary_path = self.save_summary(final_metrics)

        print('================ Final Metrics ================')
        print(f'Best epoch: {self.best_epoch}')
        print(
            f'Train | MSE={final_metrics["train"]["mse"]:.6f} '
            f'MAE={final_metrics["train"]["mae"]:.6f} '
            f'Corr={final_metrics["train"]["corr"]:.4f}'
        )
        print(
            f'Val   | MSE={final_metrics["val"]["mse"]:.6f} '
            f'MAE={final_metrics["val"]["mae"]:.6f} '
            f'Corr={final_metrics["val"]["corr"]:.4f}'
        )
        print(
            f'Test  | MSE={final_metrics["test"]["mse"]:.6f} '
            f'MAE={final_metrics["test"]["mae"]:.6f} '
            f'Corr={final_metrics["test"]["corr"]:.4f}'
        )
        print(f'模型與結果已輸出到: {self.out_dir}')

        return {
            'best_path': self.best_path,
            'history_csv': history_csv,
            'summary_json': summary_path,
            'final_metrics': final_metrics,
        }