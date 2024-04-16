import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import POIRecommender
from data import POIDataModule

def main(args):
    # Set up data module
    data_module = POIDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    data_module.prepare_data()

    # Load model checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    model = POIRecommender.load_from_checkpoint(checkpoint_path)

    # Set up trainer
    trainer = pl.Trainer(gpus=args.gpus)

    # Evaluate on test data
    test_results = trainer.test(model=model, dataloaders=data_module.test_dataloader())

    # Compute additional metrics
    test_preds = test_results[0]['test/preds']
    test_labels = test_results[0]['test/labels']

    accuracy = accuracy_score(test_labels.cpu(), test_preds.cpu())
    precision = precision_score(test_labels.cpu(), test_preds.cpu(), average='macro', zero_division='raise')
    recall = recall_score(test_labels.cpu(), test_preds.cpu(), average='macro', zero_division='raise')
    f1 = f1_score(test_labels.cpu(), test_preds.cpu(), average='macro', zero_division='raise')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default='best_model.ckpt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpus', type=int, default=0)

    args = parser.parse_args()

    main(args)