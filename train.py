import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from model import POIRecommender
from data import POIDataModule

def main(args):
    # Set up data module
    data_module = POIDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    data_module.prepare_data()

    # Set up model
    num_users = data_module.train_data['user_id'].nunique()
    num_pois = data_module.train_data['poi_id'].nunique()
    model = POIRecommender(num_users=num_users, num_pois=num_pois)

    # Set up loggers and callbacks
    logger = TensorBoardLogger(save_dir=args.log_dir)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.checkpoint_dir,
        filename='poi-recommender-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min'
    )

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    # Train the model
    trainer.fit(model, data_module)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--gpus', type=int, default=0)

    args = parser.parse_args()

    main(args)