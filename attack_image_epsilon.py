import torch
import numpy as np
from torch.autograd import Variable
from data import get_task_data_in_numpy, NeuroDataModule
from model import SegmentationModule
from utils import visualize_attack_results
import matplotlib.pyplot as plt


# class EpsilonImageAttack:
#     def __init__(self, model):
#         self.model = model
#         self.epsilon = torch.zeros((256, 256))
#
#     def epsilon_attack(self, x, y, alpha=0.01, max_iter=300, clipping=0.1):
#         self.model.eval()
#         x_fooling = x.clone()
#         y_target = torch.where(y == 1, 2, y)
#
#         ones = (y_target == 1).sum()
#         print("target label", ones)
#
#         epsilon = torch.zeros_like(x)
#         x_fooling_var = Variable(x_fooling, requires_grad=True)
#
#         for it in range(max_iter):
#             attack_image = x_fooling_var + epsilon
#             score_model = self.model(attack_image)
#             loss = self.model.loss(score_model, y_target)
#             loss.backward()
#
#             gradient = x_fooling_var.grad.data
#             epsilon -= torch.sign(gradient) * alpha
#             epsilon = torch.clamp(epsilon, min=-clipping, max=clipping)
#             x_fooling_var.grad.fill_(0)
#
#         self.epsilon = epsilon
#         return epsilon


def epsilon_attack(model, x, y):
    alpha = 0.01  # learning rate is 1
    max_iter = 500  # maximum number of iterations
    clipping = 0.1

    model.eval()
    x_fooling = x.clone()
    y_target = torch.where(y == 1, 2, y)

    ones = (y_target == 1).sum()
    print("target label", ones)

    epsilon = torch.zeros_like(x)
    x_fooling_var = Variable(x_fooling, requires_grad=True)

    for it in range(max_iter):
        attack_image = x_fooling_var + epsilon
        score_model = model(attack_image)
        loss = model.loss(score_model, y_target)
        loss.backward()

        gradient = x_fooling_var.grad.data
        epsilon -= torch.sign(gradient) * alpha
        epsilon = torch.clamp(epsilon, min=-clipping, max=clipping)
        x_fooling_var.grad.fill_(0)

    return epsilon

def run_attack():
    checkpoint_path = f'version_3/checkpoints/epoch=99-step=1700.ckpt'
    model = SegmentationModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    dataloader = NeuroDataModule(32).test_dataloader()
    # attack = Attack()

    for idx, batch in enumerate(dataloader):
        if idx == 0:
            x, y = batch
            x = x[0].unsqueeze(0)
            y = y[0].unsqueeze(0)
            y_pred = model.predict(x)

            # X_fooling, perturbation = attack.make_perturbation(x, y, 0, model)

            perturbation = epsilon_attack(model, x, y)
            X_fooling = x + perturbation

            y_attack_pred = model.predict(X_fooling)

            correct_ones = (y_pred == 1).sum()
            attack_ones = (y_attack_pred == 1).sum()
            print("predx", correct_ones)
            print("pred_attack", attack_ones)

            visualize_attack_results("image_epsilon", x, X_fooling, perturbation, y, y_pred, y_attack_pred)
            break


if __name__ == "__main__":
    run_attack()
