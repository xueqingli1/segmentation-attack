import torch


def generate_hide_one_label(truth_y, hide_label, target_label):
    """
    Generates target labels by changing hide_label to target_label
    Original labels: 0:Background; 1:Vessels; 2:Cells; 3:Axons
    """
    target_y = torch.where(truth_y == hide_label, target_label, truth_y)
    return target_y


def hide_with_nearest_segment(y_true, hide_label):
    """
        Changes each pixel of y_true with hide_label to the label of the nearest pixel with a different label
        input: y_true: tensor of shape (height, width)
        output: y_true: tensor of shape (height, width)
    """
    y_target = torch.clone(y_true)
    # y_target = y_target.squeeze(0)
    # y_true = y_true.squeeze(0)

    print("before replacing", count_label(y_target))
    print("y_true", y_target.shape)
    for batch in range(y_target.shape[0]):
        for i in range(y_target.shape[1]):
            for j in range(y_target.shape[2]):
                if y_target[batch, i, j] == hide_label:
                    closest = 256 ** 2 + 256 ** 2
                    label = 0
                    if hide_label == 0:
                        label = 3
                    search_left = max(0, i - 25)
                    search_right = min(256, i + 25)
                    search_top = max(0, j - 25)
                    search_bottom = min(256, j + 25)
                    for r in range(search_left, search_right):
                        for c in range(search_top, search_bottom):
                            dist = (i - r) ** 2 + (j - c) ** 2
                            if dist < closest and y_true[batch, r, c] != y_target[batch, i, j]:
                                closest = dist
                                label = y_true[batch, r, c]
                    y_target[batch, i, j] = label

    # y_target = y_target.unsqueeze(0)
    print("y_true", y_target.shape)
    print("after replacing", count_label(y_target))
    return y_target


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


