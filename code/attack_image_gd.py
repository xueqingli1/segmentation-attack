import torch
import numpy as np
from torch.autograd import Variable
from data import get_task_data_in_numpy, NeuroDataModule
from model import SegmentationModule
from utils import visualize_attack_results
import matplotlib.pyplot as plt
from labels import generate_random_label, generate_hide_one_label, count_label
from labels2 import hide_with_nearest_segment


class GradientDescentImageAttack:
    def make_perturbation(self, model, x, y_target, learning_rate=0.5, max_iter=100, clipping=0.2):
        """
        Generates a perturbation
        """
        model.eval()
        X_fooling = x.clone()
        X_fooling_var = Variable(X_fooling, requires_grad=True)
        perturbation = torch.zeros_like(x)

        for it in range(max_iter):
            attack_image = X_fooling_var + perturbation
            attack_image = torch.clamp(attack_image, min=0, max=1)
            score_model = model(attack_image)
            loss = model.loss(score_model, y_target)
            loss.backward()

            gradient = X_fooling_var.grad.data
            norm = torch.sqrt(torch.sum(gradient ** 2))
            perturbation -= learning_rate * gradient / norm
            perturbation = torch.clamp(perturbation, min=-clipping, max=clipping)
            X_fooling_var.grad.fill_(0)

        return perturbation


def run_attack():
    checkpoint_path = f'../version_3/checkpoints/epoch=99-step=1700.ckpt'
    model = SegmentationModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    dataloader = NeuroDataModule(32).test_dataloader()
    attack = GradientDescentImageAttack()

    for idx, batch in enumerate(dataloader):
        if idx == 3:
            x, y = batch
            x = x[1].unsqueeze(0)
            y = y[1].unsqueeze(0)
            y_pred = model.predict(x)

            for hide_label in range(4):
                y_target = hide_with_nearest_segment(y, hide_label)
                perturbation = attack.make_perturbation(model, x, y_target, learning_rate=0.5, max_iter=100)
                X_fooling = x + perturbation
                X_fooling = torch.clamp(X_fooling, min=0, max=1)
                y_attack_pred = model.predict(X_fooling)
                print("hide label", hide_label)
                print("before attack", count_label(y_pred))
                print("after attack", count_label(y_attack_pred))

                visualize_attack_results(f"nn_clip_{hide_label}_image_gd",
                                         x, X_fooling, perturbation, y, y_target, y_attack_pred)


if __name__ == "__main__":
    run_attack()
