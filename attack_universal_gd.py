import torch
import numpy as np
from torch.autograd import Variable
from data import NeuroDataModule
from model import SegmentationModule
from utils import visualize_attack_results


class GradientDescentUniversalAttack:
    def __init__(self, model, dataset=NeuroDataModule(32)):
        self.model = model
        self.train_dataloader = dataset.train_dataloader()
        self.test_dataloader = dataset.test_dataloader()
        self.epsilon = torch.zeros((256, 256))

    def attack_train(self, alpha=0.01, max_iter=100, clipping=0.1):
        self.model.eval()

        epsilon = torch.zeros((1, 256, 256))
        for it in range(max_iter):
            aggregated_gradient = torch.zeros((1, 256, 256))
            for idx, batch in enumerate(self.train_dataloader):
                x, y = batch
                y_target = torch.where(y == 1, 2, y)
                batch_size = x.shape[0]
                print(x.shape)
                print("batch_size", batch_size)
                epsilon_batch = torch.tile(epsilon, (batch_size, 1, 1, 1))

                x_fooling = x.clone()
                x_fooling_var = Variable(x_fooling, requires_grad=True)

                attacked_images = x_fooling_var + epsilon_batch
                score_model = self.model(attacked_images)
                loss = self.model.loss(score_model, y_target)
                loss.backward()

                gradient = x_fooling_var.grad.data
                print("gradient shape", gradient.shape)
                sum_gradient = torch.sum(gradient, 0)
                print("sum_gradient shape", sum_gradient.shape)
                print("aggregated_gradient shape", aggregated_gradient.shape)
                aggregated_gradient += sum_gradient

                x_fooling_var.grad.fill_(0)

            epsilon -= torch.sign(aggregated_gradient) * alpha
            epsilon = torch.clamp(epsilon, min=-clipping, max=clipping)
            print("epsilon", epsilon)

        self.epsilon = epsilon
        return epsilon

    def attack_test(self):
        for idx, batch in enumerate(self.test_dataloader):
            if idx == 0:
                x, y = batch
                x = x[0].unsqueeze(0)
                y = y[0].unsqueeze(0)
                y_pred = model.predict(x)

                X_fooling = x + self.epsilon

                y_attack_pred = model.predict(X_fooling)

                correct_ones = (y_pred == 1).sum()
                attack_ones = (y_attack_pred == 1).sum()
                print("predx", correct_ones)
                print("pred_attack", attack_ones)

                visualize_attack_results("universal", x, X_fooling, self.epsilon, y, y_pred, y_attack_pred)
                break


if __name__ == "__main__":
    checkpoint_path = f'version_3/checkpoints/epoch=99-step=1700.ckpt'
    model = SegmentationModule.load_from_checkpoint(checkpoint_path)
    dataset = NeuroDataModule(32)
    uni_attack = EpsilonUniversalAttack(model, dataset)
    uni_attack.attack_train(alpha=0.01, max_iter=100, clipping=0.1)
    uni_attack.attack_test()

