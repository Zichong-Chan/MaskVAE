import numpy as np
import torch
import torch.nn.functional as f


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 19:  # CelebAMask-HQ
        cmap = np.array([(0, 0, 0), (204, 0, 0), (76, 153, 0),
                         (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                         (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                         (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
                         (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class Colorize(object):
    def __init__(self, n=19):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def tensor2label(label_tensor, n_label, imtype=np.uint8):
    # if n_label == 0:
    #     return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    # label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    label_numpy = label_tensor.numpy()
    label_numpy = label_numpy / 255.0

    return label_numpy


def generate_label(inputs, imsize):
    pred_batch = []
    for input in inputs:
        input = input.view(1, 19, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)

    label_batch = []
    for p in pred_batch:
        p = p.view(1, imsize, imsize)
        label_batch.append(tensor2label(p, 19))

    label_batch = np.array(label_batch)
    label_batch = torch.from_numpy(label_batch)

    return label_batch


def convert_onehot(inputs, num_classes=19, device='cuda'):
    # inputs = torch.from_numpy(inputs)
    b, h, w = inputs.shape
    return f.one_hot(inputs, num_classes=num_classes).view(b, num_classes, h, w).float().to(device)


def labels_colorize(labels_plain):
    colors = []
    for idx in range(labels_plain.shape[0]):
        colors.append(torch.from_numpy((Colorize()(labels_plain[idx]).numpy())/255.))
    colors = torch.stack(colors, dim=0)
    return colors



