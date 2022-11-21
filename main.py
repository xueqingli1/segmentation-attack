import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from model import SegmentationModule
from data import NeuroDataModule
from utils import draw_mask_comparsion, save_result
from config import *
import numpy as np
import matplotlib.pyplot as plt
import torch

torch.manual_seed(42)
np.random.seed(42)

def train(model, loss_type, encoder, encoder_weights, step_lr, gpu_id):
    xd = NeuroDataModule(64)
    model = SegmentationModule(model, loss_type, encoder, encoder_weights, step_lr)
    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=False)
    EPOCH = 100
    if DEBUG:
        train_trainer = pl.Trainer(accelerator='cpu',
                                    # devices=[gpu_id],
                                    auto_select_gpus=True,
                                    max_epochs=EPOCH, 
                                    check_val_every_n_epoch=5,
                                    log_every_n_steps=10,
                                    )
    else:
        train_trainer = pl.Trainer(accelerator='cpu',
                                    # devices=6,
                                    max_epochs=EPOCH, 
                                    strategy=ddp,
                                    check_val_every_n_epoch=5,
                                    log_every_n_steps=10,
                                    )
    train_trainer.fit(model=model, datamodule=xd)

def train_one(checkpoint_path):
    xd = NeuroDataModule(32)
    model = SegmentationModule.load_from_checkpoint(checkpoint_path)
    test_trainer = pl.Trainer(accelerator='cpu',
                                    # devices=1,
                                    max_epochs=1, 
                                    reload_dataloaders_every_n_epochs=2,
                                    log_every_n_steps=10,
                                    )
    test_trainer.fit(model=model, datamodule=xd)

def test(checkpoint_path):
    xd = NeuroDataModule(32)
    model = SegmentationModule.load_from_checkpoint(checkpoint_path)
    test_trainer = pl.Trainer(accelerator='cpu',
                                    devices=1,
                                    max_epochs=1, 
                                    check_val_every_n_epoch=2,
                                    reload_dataloaders_every_n_epochs=2,
                                    log_every_n_steps=10,
                                    )
    test_trainer.test(model=model, datamodule=xd)

def validate(checkpoint_path):
    xd = NeuroDataModule(32)
    model = SegmentationModule.load_from_checkpoint(checkpoint_path)
    test_trainer = pl.Trainer(accelerator='cpu',
                                    devices=1,
                                    max_epochs=1, 
                                    check_val_every_n_epoch=2,
                                    reload_dataloaders_every_n_epochs=2,
                                    log_every_n_steps=10,
                                    )
    test_trainer.validate(model=model, datamodule=xd)


def visualizeResult(checkpoint_path, save_path):
    # sample_indexs = [0, 50, 100, 150, 199]
    sample_indexs = [0]
    xd = NeuroDataModule(32)
    model = SegmentationModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    dataloader = xd.neuro_test
    annotation_matrix = []
    for idx, batch in enumerate(dataloader):
        if idx not in sample_indexs:
            continue
        x, y = batch
        x = x.reshape(1, 1, 256, 256)
        y_hat = model.predict(x)
        # x = x.reshape(256, 256).numpy()
        # y = y.reshape(256, 256).numpy()
        x = x.reshape(256, 256)
        y = y.reshape(256, 256)
        y_hat = y_hat.reshape(256, 256).numpy()
        img, anno, anno_pred = draw_mask_comparsion(x, y, y_hat)
        pad_img = np.pad(img, [[1,1], [1,1], [0,0]], 'constant', constant_values=1.0)
        pad_anno = np.pad(anno, [[1,1], [1,1], [0,0]], 'constant', constant_values=1.0)
        pad_anno_pred = np.pad(anno_pred, [[1,1], [1,1], [0,0]], 'constant', constant_values=1.0)
        annotation_matrix.append(np.concatenate([pad_img, pad_anno, pad_anno_pred], axis=1))
    annotation_matrix = np.concatenate(annotation_matrix, axis=0)
    plt.figure(figsize=(len(sample_indexs) * 15, 3 * 15))
    plt.imshow(annotation_matrix)
    plt.axis('off')
    plt.savefig(f'combined_version{save_path}.png')


if __name__ == '__main__':
    # training
    model = 'unet'
    encoder = 'resnet18'
    encoder_weights = 'imagenet'
    loss_type = 'ce'
    step_lr = None
    gpu_id = 1
    # train(model, loss_type, encoder, encoder_weights, step_lr, gpu_id)
    # validation, testing and visualization
    # for i in [3]:
    #     print('=' * 20, f'version {i}', '=' * 20)
    #     checkpoint_path = f'lightning_logs/version_{i}/checkpoints/epoch=99-step=1700.ckpt'
    #     validate(checkpoint_path)
    #     test(checkpoint_path)
    #     visualizeResult(checkpoint_path, i)
    checkpoint_path = f'version_3/checkpoints/epoch=99-step=1700.ckpt'
    # validate(checkpoint_path)
    # test(checkpoint_path)
    visualizeResult(checkpoint_path, 3)