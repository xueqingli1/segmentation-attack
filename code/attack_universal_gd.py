import torch
import numpy as np
from torch.autograd import Variable
from data import NeuroDataModule
from model import SegmentationModule
from utils import visualize_attack_results
from labels import generate_random_label, generate_hide_one_label, count_label
from labels2 import hide_with_nearest_segment

class GradientDescentUniversalAttack:
    def __init__(self, model, dataset=NeuroDataModule(32)):
        self.model = model
        self.train_dataloader = dataset.train_dataloader()
        self.test_dataloader = dataset.test_dataloader()
        self.epsilon = torch.zeros((1, 256, 256))
        print(len(self.train_dataloader))

    def attack_train(self, learning_rate=0.1, max_iter=10, clipping=0.1):
        self.model.eval()

        epsilon = torch.zeros((1, 256, 256))
        for it in range(max_iter):
            print("iteration", it)
            aggregated_gradient = torch.zeros((1, 256, 256))
            for idx, batch in enumerate(self.train_dataloader):
                if idx % 2 == 1:
                    continue
                x, y = batch
                # y_target = torch.where(y == 1, 2, y)
                y_target = hide_with_nearest_segment(y, 2)
                batch_size = x.shape[0]
                epsilon_batch = torch.tile(epsilon, (batch_size, 1, 1, 1))

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

            norm = torch.sqrt(torch.sum(aggregated_gradient ** 2))
            epsilon -= learning_rate * aggregated_gradient / norm
            # epsilon = torch.clamp(epsilon, min=-clipping, max=clipping)
            print("epsilon", epsilon)

        self.epsilon = epsilon
        return epsilon

    def attack_test(self):
        for idx, batch in enumerate(self.test_dataloader):
            if idx == 0:
                x, y = batch
                x = x[0].unsqueeze(0)
                y = y[0].unsqueeze(0)
                y_pred = self.model.predict(x)

                X_fooling = x + self.epsilon
                y_attack_pred = self.model.predict(X_fooling)

                print("Gradient Descent Universal Attack")
                print("before attack", count_label(y_pred))
                print("after attack", count_label(y_attack_pred))

                visualize_attack_results("universal_gd", x, X_fooling, self.epsilon, y, y_pred, y_attack_pred)
                break


if __name__ == "__main__":
    checkpoint_path = f'../version_3/checkpoints/epoch=99-step=1700.ckpt'
    model = SegmentationModule.load_from_checkpoint(checkpoint_path)
    dataset = NeuroDataModule(32)
    uni_attack = GradientDescentUniversalAttack(model, dataset)
    uni_attack.attack_train(learning_rate=0.1, max_iter=5, clipping=0.2)
    uni_attack.attack_test()
