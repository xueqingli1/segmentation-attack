import torch
import numpy as np
from torch.autograd import Variable
from data import get_task_data_in_numpy, NeuroDataModule
from model import SegmentationModule
from utils import visualize_attack_results
import matplotlib.pyplot as plt
from labels import generate_random_label, generate_hide_one_label, count_label


class GradientDescentImageAttack:
    def make_perturbation(self, model, x, y_target, learning_rate=0.01, max_iter=500):
        """
        Generates a perturbation
        """
        model.eval()
        X_fooling = x.clone()
        X_fooling_var = Variable(X_fooling, requires_grad=True)

        for it in range(max_iter):
            score_model = model(X_fooling_var)
            loss = model.loss(score_model, y_target)
            loss.backward()

            gradient = X_fooling_var.grad.data
            norm = torch.sqrt(torch.sum(gradient ** 2))
            X_fooling_var.data -= learning_rate * gradient / norm
            X_fooling_var.grad.fill_(0)
            X_fooling_var.data = torch.clamp(X_fooling_var.data, min=0, max=1)

        X_fooling = X_fooling_var.data
        perturbation = X_fooling - x
        return X_fooling, perturbation


def run_attack():
    checkpoint_path = f'../version_3/checkpoints/epoch=99-step=1700.ckpt'
    model = SegmentationModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    dataloader = NeuroDataModule(32).test_dataloader()
    attack = GradientDescentImageAttack()

    for idx, batch in enumerate(dataloader):
        if idx == 2:
            x, y = batch
            x = x[0].unsqueeze(0)
            y = y[0].unsqueeze(0)
            y_pred = model.predict(x)
            y_target = torch.where(y == 1, 2, y)

            X_fooling, perturbation = attack.make_perturbation(model, x, y_target, learning_rate=0.01, max_iter=1)
            y_attack_pred = model.predict(X_fooling)

            print("before attack", count_label(y_pred))
            print("after attack", count_label(y_attack_pred))

            visualize_attack_results("image_gd", x, X_fooling, perturbation, y, y_pred, y_attack_pred)
            break


if __name__ == "__main__":
    run_attack()
