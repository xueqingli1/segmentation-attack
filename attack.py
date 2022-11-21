import torch
import numpy as np
from torch.autograd import Variable
from data import get_task_data_in_numpy, NeuroDataModule
from model import SegmentationModule
from utils import visualize_attack_results
import matplotlib.pyplot as plt


class Attack:
    def hide_one_label(self, truth_y, hide_label, target_label):
        """
        Generates target labels without hide_label
        Original labels: 0:Background; 1:Vessels; 2:Cells; 3:Axons

        Inputs:
        - truth_y: image label Tensor of shape (1, 256, 256)
        - hide_label: An integer in the range [0, 3]
        - target_label: An integer in the range [0, 3]

        Returns:
        - Y_attacked: Output labels similar to X but without segment_y
        """
        target_y = torch.where(truth_y == hide_label, target_label, truth_y)
        return target_y

    def generate_random_label(self, truth_y, hide_label, target_label):
        target_y = torch.where(truth_y == hide_label, target_label, truth_y)
        return target_y

    def make_perturbation(self, X, Y, segment_y, model):
        """
        Generates a perturbation which makes the model not able to 
        detect segment_y.

        Inputs:
        - X: Input image; Tensor of shape (1, 256, 256)
        - segment_y: An integer in the range [0, 3)
        - model: A pretrained segmentatino model

        Returns:
        - X_attacked: An image that is close to X, but that is hides segment segment_y
        by the model.
        """

        model.eval()

        X_fooling = X.clone()

        # generate different type of attack labels

        # Y_target = generate_attack_label(self, truth_y, hide_label, target_label)
        Y_target = torch.where(Y == 1, 2, Y)

        ones = (Y_target == 1).sum()
        print("target label", ones)
        # epsilon = torch.zeros_like(X)
        # epsiolon_var = Variable(epsilon, requires_grad=True)
        # X_var = Variable(X, requires_grad=True)
        X_fooling_var = Variable(X_fooling, requires_grad=True)

        learning_rate = 0.01  # learning rate is 1
        max_iter = 500  # maximum number of iterations

        for it in range(max_iter):
            score_model = model(X_fooling_var)
            # pred_idx = torch.argmax(score_model, dim = 1)
            loss = model.loss(score_model, Y_target)
            loss.backward()

            gradient = X_fooling_var.grad.data
            norm = torch.sqrt(torch.sum(gradient ** 2))
            X_fooling_var.data -= learning_rate * gradient / norm
            X_fooling_var.grad.fill_(0)
            X_fooling_var.data = torch.clamp(X_fooling_var.data, min=0, max=1)

        X_fooling = X_fooling_var.data
        diff = X_fooling - X
        diff = diff.flatten()
        mindiff = min(diff)

        perturbation = X_fooling - X
        perturbation = torch.sub(perturbation, mindiff)

        return X_fooling, perturbation


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
    attack = Attack()

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

            visualize_attack_results("image_dependent", x, X_fooling, perturbation, y, y_pred, y_attack_pred)
            break


if __name__ == "__main__":
    run_attack()






