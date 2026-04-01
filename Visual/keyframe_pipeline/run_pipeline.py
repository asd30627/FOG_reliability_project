import argparse
from pathlib import Path

from extract_features import ReliabilityFeatureExtractor
from build_reliability_dataset import ReliabilityDatasetBuilder
from train_reliability_model import ReliabilityModelTrainer
from infer_reliability import ReliabilityInferencer


class ReliabilityPipelineRunner:
    def __init__(
        self,
        sequence_dir,
        # 共用
        feature_csv='',
        dataset_dir='',
        checkpoint='',
        inference_out_dir='',
        # stage 開關
        run_feature_extraction=True,
        run_dataset_build=True,
        run_training=True,
        run_inference=True,
        # feature extractor
        cache_csv_path='',
        feature_out_dir='',
        clean_feature_output_dir=False,
        # dataset builder
        seq_len=8,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        min_rows=16,
        use_validity_mask=False,
        # trainer
        trainer_out_dir='',
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
        # inferencer
        tau=0.5,
        infer_batch_size=256,
    ):
        self.sequence_dir = Path(sequence_dir).expanduser().resolve()

        self.run_feature_extraction = bool(run_feature_extraction)
        self.run_dataset_build = bool(run_dataset_build)
        self.run_training = bool(run_training)
        self.run_inference = bool(run_inference)

        self.feature_csv = Path(feature_csv).expanduser().resolve() if feature_csv else self.sequence_dir / 'features' / 'local_visual_features.csv'
        self.dataset_dir = Path(dataset_dir).expanduser().resolve() if dataset_dir else self.sequence_dir / 'reliability_dataset'
        self.checkpoint = Path(checkpoint).expanduser().resolve() if checkpoint else self.dataset_dir / 'model_runs' / model_type / 'best.pt'
        self.inference_out_dir = Path(inference_out_dir).expanduser().resolve() if inference_out_dir else self.sequence_dir / 'reliability_inference'

        self.cache_csv_path = Path(cache_csv_path).expanduser().resolve() if cache_csv_path else self.sequence_dir / 'keyframes' / 'accepted_pairs_cache.csv'
        self.feature_out_dir = Path(feature_out_dir).expanduser().resolve() if feature_out_dir else self.sequence_dir / 'features'
        self.clean_feature_output_dir = bool(clean_feature_output_dir)

        self.seq_len = int(seq_len)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.min_rows = int(min_rows)
        self.use_validity_mask = bool(use_validity_mask)

        self.trainer_out_dir = Path(trainer_out_dir).expanduser().resolve() if trainer_out_dir else self.dataset_dir / 'model_runs' / model_type
        self.model_type = str(model_type)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.seed = int(seed)
        self.device = device

        self.tau = float(tau)
        self.infer_batch_size = int(infer_batch_size)

    def run_feature_stage(self):
        print('========== Stage 1/4: Feature Extraction ==========')
        extractor = ReliabilityFeatureExtractor(
            sequence_dir=str(self.sequence_dir),
            cache_csv_path=str(self.cache_csv_path),
            out_dir=str(self.feature_out_dir),
            clean_output_dir=self.clean_feature_output_dir,
        )
        result = extractor.run()
        return result

    def run_dataset_stage(self):
        print('========== Stage 2/4: Build Reliability Dataset ==========')
        builder = ReliabilityDatasetBuilder(
            sequence_dir=str(self.sequence_dir),
            feature_csv=str(self.feature_csv),
            out_dir=str(self.dataset_dir),
            seq_len=self.seq_len,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            min_rows=self.min_rows,
            use_validity_mask=self.use_validity_mask,
        )
        result = builder.build()
        return result

    def run_training_stage(self):
        print('========== Stage 3/4: Train Reliability Model ==========')
        trainer = ReliabilityModelTrainer(
            dataset_dir=str(self.dataset_dir),
            out_dir=str(self.trainer_out_dir),
            model_type=self.model_type,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            seed=self.seed,
            device=self.device,
        )
        result = trainer.run()
        return result

    def run_inference_stage(self):
        print('========== Stage 4/4: Infer Reliability ==========')
        inferencer = ReliabilityInferencer(
            sequence_dir=str(self.sequence_dir),
            feature_csv=str(self.feature_csv),
            dataset_dir=str(self.dataset_dir),
            checkpoint=str(self.checkpoint),
            out_dir=str(self.inference_out_dir),
            tau=self.tau,
            batch_size=self.infer_batch_size,
            device=self.device,
        )
        result = inferencer.run()
        return result

    def run(self):
        summary = {}

        if self.run_feature_extraction:
            summary['feature_extraction'] = self.run_feature_stage()
        else:
            print('========== Stage 1/4: Feature Extraction (skip) ==========')

        if self.run_dataset_build:
            summary['dataset_build'] = self.run_dataset_stage()
        else:
            print('========== Stage 2/4: Build Reliability Dataset (skip) ==========')

        if self.run_training:
            summary['training'] = self.run_training_stage()
            # 若 training 有跑，checkpoint 改用剛訓練出來的 best.pt
            best_path = summary['training'].get('best_path', None)
            if best_path is not None:
                self.checkpoint = Path(best_path)
        else:
            print('========== Stage 3/4: Train Reliability Model (skip) ==========')

        if self.run_inference:
            summary['inference'] = self.run_inference_stage()
        else:
            print('========== Stage 4/4: Infer Reliability (skip) ==========')

        print('========== Pipeline Finished ==========')
        for stage_name, stage_result in summary.items():
            print(f'[{stage_name}]')
            if isinstance(stage_result, dict):
                for k, v in stage_result.items():
                    print(f'  {k}: {v}')
            else:
                print(f'  {stage_result}')

        return summary


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sequence_dir', type=str, required=True, help='例如: /mnt/sata4t/dataset/sequence_001')

    parser.add_argument('--feature_csv', type=str, default='', help='若留空，預設 sequence_dir/features/local_visual_features.csv')
    parser.add_argument('--dataset_dir', type=str, default='', help='若留空，預設 sequence_dir/reliability_dataset')
    parser.add_argument('--checkpoint', type=str, default='', help='若留空，預設 dataset_dir/model_runs/<model_type>/best.pt')
    parser.add_argument('--inference_out_dir', type=str, default='', help='若留空，預設 sequence_dir/reliability_inference')

    parser.add_argument('--cache_csv_path', type=str, default='', help='若留空，預設 sequence_dir/keyframes/accepted_pairs_cache.csv')
    parser.add_argument('--feature_out_dir', type=str, default='', help='若留空，預設 sequence_dir/features')
    parser.add_argument('--trainer_out_dir', type=str, default='', help='若留空，預設 dataset_dir/model_runs/<model_type>')

    parser.add_argument('--skip_feature_extraction', action='store_true')
    parser.add_argument('--skip_dataset_build', action='store_true')
    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--skip_inference', action='store_true')

    parser.add_argument('--clean_feature_output_dir', action='store_true')

    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--train_ratio', type=float, default=0.70)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--min_rows', type=int, default=16)
    parser.add_argument('--use_validity_mask', action='store_true')

    parser.add_argument('--model_type', type=str, default='gru', choices=['mlp', 'gru', 'tcn'])
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if __import__('torch').cuda.is_available() else 'cpu')

    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--infer_batch_size', type=int, default=256)

    return parser.parse_args()


def main():
    args = parse_args()

    runner = ReliabilityPipelineRunner(
        sequence_dir=args.sequence_dir,
        feature_csv=args.feature_csv,
        dataset_dir=args.dataset_dir,
        checkpoint=args.checkpoint,
        inference_out_dir=args.inference_out_dir,

        run_feature_extraction=not args.skip_feature_extraction,
        run_dataset_build=not args.skip_dataset_build,
        run_training=not args.skip_training,
        run_inference=not args.skip_inference,

        cache_csv_path=args.cache_csv_path,
        feature_out_dir=args.feature_out_dir,
        clean_feature_output_dir=args.clean_feature_output_dir,

        seq_len=args.seq_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_rows=args.min_rows,
        use_validity_mask=args.use_validity_mask,

        trainer_out_dir=args.trainer_out_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,

        tau=args.tau,
        infer_batch_size=args.infer_batch_size,
    )
    runner.run()


if __name__ == '__main__':
    main()