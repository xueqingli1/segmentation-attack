import torch
import numpy as np
from torch.autograd import Variable
from data import get_task_data_in_numpy, NeuroDataModule
from model import SegmentationModule
from utils import visualize_attack_results
import matplotlib.pyplot as plt
from labels import generate_random_label, generate_hide_one_label, count_label
from labels2 import hide_with_nearest_segment

class EpsilonImageAttack:
    def __init__(self, model):
        self.model = model
        self.epsilon = torch.zeros((256, 256))

    def epsilon_attack(self, x, y_target, alpha=0.01, max_iter=500, clipping=0.1):
        self.model.eval()
        x_fooling = x.clone()
        # y_target = torch.where(y == 1, 2, y)

        epsilon = torch.zeros_like(x)
        x_fooling_var = Variable(x_fooling, requires_grad=True)

        for it in range(max_iter):
            attack_image = x_fooling_var + epsilon
            attack_image = torch.clamp(attack_image, min=0, max=1)
            score_model = self.model(attack_image)
            loss = self.model.loss(score_model, y_target)
            loss.backward()

            gradient = x_fooling_var.grad.data
            epsilon -= torch.sign(gradient) * alpha
            epsilon = torch.clamp(epsilon, min=-clipping, max=clipping)
            x_fooling_var.grad.fill_(0)

        self.epsilon = epsilon
        return epsilon


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
    checkpoint_path = f'../version_3/checkpoints/epoch=99-step=1700.ckpt'
    model = SegmentationModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    dataloader = NeuroDataModule(32).test_dataloader()
    attack = EpsilonImageAttack(model)

    for idx, batch in enumerate(dataloader):
        if idx == 3:
            x, y = batch
            x = x[1].unsqueeze(0)
            y = y[1].unsqueeze(0)
            print(y.shape)
            y_pred = model.predict(x)

            for hide_label in range(4):
                y_target = hide_with_nearest_segment(y, hide_label)
                perturbation = attack.epsilon_attack(x, y_target, alpha=0.01, max_iter=200, clipping=0.1)

                X_fooling = x + perturbation
                X_fooling = torch.clamp(X_fooling, min=0, max=1)

                y_attack_pred = model.predict(X_fooling)
                print("hide label", hide_label)
                print("before attack", count_label(y_pred))
                print("after attack", count_label(y_attack_pred))
                visualize_attack_results(f"nn_hide_{hide_label}_image_epsilon",
                                         x, X_fooling, perturbation, y, y_pred, y_attack_pred)

            # for hide_label in range(4):
            #     for target_label in range(4):
            #         if target_label != hide_label:
            #             y_target = generate_hide_one_label(y, hide_label, target_label)
            #
            #             perturbation = attack.epsilon_attack(x, y_target, alpha=0.01, max_iter=200, clipping=0.1)
            #
            #             X_fooling = x + perturbation
            #             y_attack_pred = model.predict(X_fooling)
            #             print("hide label", hide_label, "target label", target_label)
            #             print("before attack", count_label(y_pred))
            #             print("after attack", count_label(y_attack_pred))
            #
            #             visualize_attack_results(f"hide_{hide_label}_target_{target_label}_image_epsilon",
            #                                      x, X_fooling, perturbation, y, y_pred, y_attack_pred)


if __name__ == "__main__":
    run_attack()
