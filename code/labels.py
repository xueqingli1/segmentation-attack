import torch


def generate_hide_one_label(truth_y, hide_label, target_label):
    """
    Generates target labels by changing hide_label to target_label
    Original labels: 0:Background; 1:Vessels; 2:Cells; 3:Axons
    """
    target_y = torch.where(truth_y == hide_label, target_label, truth_y)
    return target_y


def generate_random_label(truth_y):
    """
    Generates random target labels
    Original labels: 0:Background; 1:Vessels; 2:Cells; 3:Axons
    """
    target_y = torch.randint(0, 4, truth_y.shape)
    print(count_label(target_y))
    return target_y


def count_label(label):
    counts = dict()
    counts[0] = (label == 0).sum()  # background
    counts[1] = (label == 1).sum()  # vessels
    counts[2] = (label == 2).sum()  # cells
    counts[3] = (label == 3).sum()  # axons
    return counts

