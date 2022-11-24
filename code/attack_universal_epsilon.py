import torch
import numpy as np
from torch.autograd import Variable
from data import NeuroDataModule
from model import SegmentationModule
from utils import visualize_attack_results
from labels import generate_random_label, generate_hide_one_label, count_label
from labels2 import hide_with_nearest_segment
import pytorch_lightning as pl
from sklearn.metrics import classification_report


class EpsilonUniversalAttack:
    def __init__(self, model, dataset=NeuroDataModule(32)):
        self.model = model
        self.train_dataloader = dataset.train_dataloader()
        self.test_dataloader = dataset.test_dataloader()
        self.epsilon = torch.zeros((1, 256, 256))
        self.data_module = dataset

    def attack_train(self, alpha=0.01, max_iter=5, clipping=0.2):
        self.model.eval()

        for it in range(max_iter):
            print("iteration", it)
            aggregated_gradient = torch.zeros((1, 256, 256))
            for idx, batch in enumerate(self.train_dataloader):
                if idx % 2 == 1:
                    continue
                x, y = batch
                # y_target = torch.where(y == 2, 1, y)
                y_target = hide_with_nearest_segment(y, 2)
                batch_size = x.shape[0]
                epsilon_batch = torch.tile(self.epsilon, (batch_size, 1, 1, 1))

                x_fooling = x.clone()
                x_fooling_var = Variable(x_fooling, requires_grad=True)

                attacked_images = x_fooling_var + epsilon_batch
                score_model = self.model(attacked_images)
                loss = self.model.loss(score_model, y_target)
                loss.backward()

                gradient = x_fooling_var.grad.data
                sum_gradient = torch.sum(gradient, 0)
                aggregated_gradient += sum_gradient
                x_fooling_var.grad.fill_(0)

            self.epsilon -= torch.sign(aggregated_gradient) * alpha
            self.epsilon = torch.clamp(self.epsilon, min=-clipping, max=clipping)
            print("epsilon", self.epsilon)
        return self.epsilon

    def generate_attack_test_images(self):
        test_loader = self.data_module.test_dataloader()
        for idx, batch in enumerate(test_loader):
            x, y = batch
            x = x[0].unsqueeze(0)
            y = y[0].unsqueeze(0)
            y_pred = self.model.predict(x)
            X_fooling = x + self.epsilon

            y_attack_pred = self.model.predict(X_fooling)

            print("Epsilon Universal Attack")
            print("before attack", count_label(y_pred))
            print("after attack", count_label(y_attack_pred))

            visualize_attack_results(f"universal_epsilon_{idx}", x, X_fooling, self.epsilon, y, y_pred, y_attack_pred)

    # def attack_test_dataset(self):
    #     self.epsilon = torch.add(self.epsilon, 0.2)
    #     y_true = []
    #     y_pred = []
    #     target_names = ['Background', 'Vessels', 'Cells', 'Axons']
    #     for idx, batch in enumerate(self.test_dataloader):
    #         x, y = batch
    #         batch_size = x.shape[0]
    #         epsilon_batch = torch.tile(self.epsilon, (batch_size, 1, 1, 1))
    #         x_fooling = x + epsilon_batch
    #         y_attack_pred = self.model.predict(x_fooling)
    #         y_true.append(y.flatten())
    #         y_pred.append(y_attack_pred.flatten())
    #     classification_report(y_true, y_pred, target_names=target_names)


if __name__ == "__main__":
    checkpoint_path = f'../version_3/checkpoints/epoch=99-step=1700.ckpt'
    model = SegmentationModule.load_from_checkpoint(checkpoint_path)
    dataset = NeuroDataModule(32)
    uni_attack = EpsilonUniversalAttack(model, dataset)
    uni_attack.attack_train(alpha=0.01, max_iter=5, clipping=0.2)
    # for i in range(10):
    #     uni_attack.attack_train(alpha=0.01, max_iter=5, clipping=0.2)
    #     torch.save(uni_attack.epsilon, f'uni_epsilon_{i}_hide_{2}.pt')
    # uni_attack.generate_attack_test_images()

