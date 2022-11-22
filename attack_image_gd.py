import torch
import numpy as np
from torch.autograd import Variable
from data import get_task_data_in_numpy, NeuroDataModule
from model import SegmentationModule
from utils import visualize_attack_results
import matplotlib.pyplot as plt


class GradientDescentImageAttack:
    def hide_one_label(self, truth_y, hide_label, target_label):
        """
        Generates target labels without hide_label
        Original labels: 0:Background; 1:Vessels; 2:Cells; 3:Axons
        """
        target_y = torch.where(truth_y == hide_label, target_label, truth_y)
        return target_y

    def make_perturbation(self, model, X, Y, learning_rate=0.01, max_iter=500):
        """
        Generates a perturbation
        """

        model.eval()
        X_fooling = X.clone()

        # Y_target = generate_attack_label(self, truth_y, hide_label, target_label)
        Y_target = torch.where(Y == 1, 2, Y)

        ones = (Y_target == 1).sum()
        print("target label", ones)

        X_fooling_var = Variable(X_fooling, requires_grad=True)

        for it in range(max_iter):
            score_model = model(X_fooling_var)
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
        # perturbation = torch.sub(perturbation, mindiff)

        return X_fooling, perturbation


def run_attack():
    checkpoint_path = f'version_3/checkpoints/epoch=99-step=1700.ckpt'
    model = SegmentationModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    dataloader = NeuroDataModule(32).test_dataloader()
    attack = GradientDescentImageAttack()

    for idx, batch in enumerate(dataloader):
        if idx == 0:
            x, y = batch
            x = x[0].unsqueeze(0)
            y = y[0].unsqueeze(0)
            y_pred = model.predict(x)

            X_fooling, perturbation = attack.make_perturbation(model, x, y, learning_rate=0.01, max_iter=500)
            y_attack_pred = model.predict(X_fooling)

            correct_ones = (y_pred == 1).sum()
            attack_ones = (y_attack_pred == 1).sum()
            print("predx", correct_ones)
            print("pred_attack", attack_ones)

            visualize_attack_results("image_dependent_gd", x, X_fooling, perturbation, y, y_pred, y_attack_pred)
            break


if __name__ == "__main__":
    run_attack()
