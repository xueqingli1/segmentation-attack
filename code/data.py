# data.py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from config import *
import matplotlib.pyplot as plt

def get_task_data_in_numpy():
    dim = DIM
    class_num = CLASS_NUM
    BASE_DIR = '../data/' + f'task2_{dim}D_{class_num}'
    images = {
        'train': np.load(BASE_DIR +'classtrainimages.npy').astype(np.uint8),
        'val': np.load(BASE_DIR +'classvalimages.npy').astype(np.uint8),
        'test': np.load(BASE_DIR +'classtestimages.npy').astype(np.uint8)
    } 
    labels = {
        'train': np.load(BASE_DIR +'classtrainlabels.npy').astype(np.uint8),
        'val': np.load(BASE_DIR +'classvallabels.npy').astype(np.uint8),
        'test': np.load(BASE_DIR +'classtestlabels.npy').astype(np.uint8)
    }
    return images, labels

class NeuroDataset(Dataset):
    def __init__(self, stage='train', transform=None):
        images, labels = get_task_data_in_numpy()
        self.data = images[stage]
        self.labels = labels[stage]
        # self.concat = np.concatenate((self.data[:, np.newaxis, :, :], self.labels[:, np.newaxis, :, :]), axis=1)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # concat = self.concat[idx].transpose(1,2,0)
        data = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            data = self.transform(data)
        return data, label
        # return concat[0, np.newaxis, :, :], (concat[1, :, :] * 255).long()

class NeuroDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        train_transforms = Compose([
            ToTensor(),
            RandomVerticalFlip(p=0.5),
            RandomHorizontalFlip(p=0.5),
            RandomRotation((0,180)),
        ])
        val_transforms = Compose([
            ToTensor(),
        ])
        self.neuro_train = NeuroDataset('train', transform=train_transforms)
        self.neuro_val = NeuroDataset('val', transform=val_transforms)
        self.neuro_test = NeuroDataset('test', transform=val_transforms)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.neuro_train, batch_size=self.batch_size, num_workers=6, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.neuro_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.neuro_test, batch_size=self.batch_size)


if __name__ == "__main__":
    images, labels = get_task_data_in_numpy()
    print("train", images['train'].shape)
    print("val", images['val'].shape)
    print("test", images['test'].shape)
    print("labels", labels['train'][0].shape)

    loader = NeuroDataModule(500).train_dataloader()
    print("loader", len(loader))

    for i, (img, mask) in enumerate(loader):
        if i == 0:
            img0 = img[0, 0].numpy()
            img1 = img[400, 0].numpy()
            mask = mask[0].numpy()
            plt.figure(figsize=(15, 5))
            plt.subplot(1,4,1)
            plt.imshow(images['train'][0], cmap='gray')
            plt.title("Original Image")
            plt.axis('off')
            plt.subplot(1,4,2)
            plt.imshow(img0, cmap='gray')
            plt.title("After Augmentation")
            plt.axis('off')

            plt.subplot(1,4,3)
            plt.imshow(images['train'][400], cmap='gray')
            plt.title("Original Image")
            plt.axis('off')
            plt.subplot(1,4,4)
            plt.imshow(img1, cmap='gray')
            plt.title("After Augmentation")
            plt.axis('off')


            plt.savefig('asdasd.png')
            break
    # for key in images.keys():
        # print(key, images[key].shape[0], np.unique(labels[key], return_counts=True))
        # print(key, np.std(images[key]), np.mean(images[key]))
        # plt.hist(images[key][0].reshape(256, -1), 256, [0, 256])
        # plt.savefig(f'{key}_hist.png')
    # plt.figure(figsize=(25, 8))
    # for i, index in enumerate([0, 200, 800, 1000]):
    #     img = images['train'][index]
    #     label = labels['train'][index]
    #     img, anno, anno_pred = draw_mask_comparsion(img, label, label)
    #     # breakpoint()
    #     plt.subplot(2, 6, i + 1)
    #     plt.imshow(img)
    #     plt.axis('off')
    #     plt.title(f'Original Image {index}')
    #     plt.subplot(2, 6, i + 1 + 6)
    #     plt.imshow(anno_pred)
    #     plt.axis('off')
    #     plt.title(f'Ground Truth {index}')
    # plt.savefig('test.png')
