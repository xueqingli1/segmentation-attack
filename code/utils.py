# utils.py
import matplotlib.pyplot as plt
import numpy as np


def draw_mask(img, mask):
    """Draws mask on an image.
    """
    mask = mask.astype(np.uint8)
    img = np.copy(img)
    h, w = img.shape[:2]
    div_factor = 255 / np.max(img)
    for i in range(h):
        for j in range(w):
            if mask[i, j] == 1:
                img[i, j] = np.array([247, 86, 124], dtype=np.float32) / div_factor
            elif mask[i, j] == 2:
                img[i, j] = np.array([153, 225, 217], dtype=np.float32) / div_factor
            elif mask[i, j] == 3:
                img[i, j] = np.array([238, 227, 171], dtype=np.float32) / div_factor
    return img

def draw_mask_comparsion(img, mask, mask_pred):
    """Draws mask on an image with ground truth and prediction.
    """
    h, w= img.shape[:2]
    img = np.repeat(img.reshape(h, w, 1), 3, axis=2)
    annotated = draw_mask(img, mask)
    annotated_pred = draw_mask(img, mask_pred)
    return (img, annotated, annotated_pred)


def draw_attack_pert_and_mask(img, pert, mask_attack_pred):
    """Draws mask on an image with ground truth and prediction.
    """
    h, w= img.shape[:2]
    img = np.repeat(img.reshape(h, w, 1), 3, axis=2)
    pert = np.repeat(pert.reshape(h, w, 1), 3, axis=2)
    annotated_pred = draw_mask(img, mask_attack_pred)
    return (img, pert, annotated_pred)

def save_result(img, annotated, annotated_pred, filename):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.subplot(1, 3, 2)
    plt.imshow(annotated)
    plt.title('Ground Truth')
    plt.subplot(1, 3, 3)
    plt.imshow(annotated_pred)
    plt.title('Prediction')
    plt.savefig(filename)

def visualize_attack_results(filename, x, x_fooling, perturbation, y, y_pred, y_attack_pred):
    x = x.reshape(256, 256).numpy()
    x_fooling = x_fooling.reshape(256, 256).numpy()
    y = y.reshape(256, 256).numpy()
    y_pred = y_pred.reshape(256, 256).numpy()
    y_attack_pred = y_attack_pred.reshape(256, 256).numpy()
    perturbation = perturbation.numpy()
    min_num = np.min(perturbation)
    perturbation -= min_num
    # max_num = np.max(perturbation)
    # perturbation /= max_num

    annotation_matrix = []
    img, anno, anno_pred = draw_mask_comparsion(x, y, y_pred)
    pad_img = np.pad(img, [[1, 1], [1, 1], [0, 0]], 'constant', constant_values=1.0)
    pad_anno = np.pad(anno, [[1, 1], [1, 1], [0, 0]], 'constant', constant_values=1.0)
    pad_anno_pred = np.pad(anno_pred, [[1, 1], [1, 1], [0, 0]], 'constant', constant_values=1.0)
    annotation_matrix.append(np.concatenate([pad_img, pad_anno, pad_anno_pred], axis=1))

    x_fooling, pert, attack_pred = draw_attack_pert_and_mask(x_fooling, perturbation, y_attack_pred)
    pad_xfool = np.pad(x_fooling, [[1, 1], [1, 1], [0, 0]], 'constant', constant_values=1.0)
    pad_pert = np.pad(pert, [[1, 1], [1, 1], [0, 0]], 'constant', constant_values=1.0)
    pad_attack_pred = np.pad(attack_pred, [[1, 1], [1, 1], [0, 0]], 'constant', constant_values=1.0)
    annotation_matrix.append(np.concatenate([pad_xfool, pad_pert, pad_attack_pred], axis=1))

    annotation_matrix = np.concatenate(annotation_matrix, axis=0)
    plt.figure(figsize=(22, 16))
    plt.suptitle(filename, fontsize=24)
    plt.imshow(annotation_matrix)
    plt.axis('off')
    plt.savefig(f'../attack_results/{filename}.png')


if __name__ == '__main__':
    images = np.load('data/task2_2D_4classtestimages.npy')
    masks = np.load('data/task2_2D_4classtestlabels.npy')
    img = images[0]
    mask = masks[0]
    mask_pred = masks[0]
    h, w= img.shape
    img = np.repeat(img.reshape(h, w, 1), 3, axis=2)
    annotated = draw_mask(img, mask)
    annotated_pred = draw_mask(img, mask_pred)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.subplot(1, 3, 2)
    plt.imshow(annotated)
    plt.title('Ground Truth')
    plt.subplot(1, 3, 3)
    plt.imshow(annotated_pred)
    plt.title('Prediction')
    plt.savefig('annotated.png')