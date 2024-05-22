import warnings

import numpy as np
import torch
import time
import torch.nn.functional as F
import random


class ContrastiveTransformations:

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, weights, biases):
        return [self.base_transforms(weights, biases) for i in range(self.n_views)]

class Compose:
    """Composes several transforms together.
    Adapted from https://pytorch.org/vision/master/_modules/torchvision/transforms/transforms.html#Compose

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, weights, biases):
        new_weights, new_biases = list(weights), list(biases)
        for t in self.transforms:
            new_weights, new_biases = t(new_weights, new_biases)
        return tuple(new_weights), tuple(new_biases)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class Translation:
    def __init__(self, translation_scale):
        self.translation_scale = translation_scale
    def __call__(self, weights, biases):
        translation = torch.empty(weights[0].shape[0]).uniform_(
            -self.translation_scale, self.translation_scale
        )
        order = random.sample(range(1, len(weights)), 1)[0]
        bias_res = translation
        i = 0
        for i in range(order):
            bias_res = bias_res @ weights[i]
        biases[i] += bias_res

        return weights, biases

class RandomTranslation:
    def __init__(self, min_scale, max_scale):
        self.min_scale = min_scale
        self.max_scale = max_scale
    def __call__(self, weights, biases):
        translation_scale = ((self.max_scale - self.min_scale) * torch.rand(1) + self.min_scale).item()
        translation = torch.empty(weights[0].shape[0]).uniform_(
            -translation_scale, translation_scale
        )
        order = random.sample(range(1, len(weights)), 1)[0]
        bias_res = translation
        i = 0
        for i in range(order):
            bias_res = bias_res @ weights[i]
        biases[i] += bias_res

        return weights, biases


class Noise:
    def __init__(self,noise_scale):
        self.noise_scale = noise_scale

    def __call__(self, weights, biases):
        new_weights = [w + w.std() * self.noise_scale for w in weights]
        new_biases = [
            b + b.std() * self.noise_scale if b.shape[0] > 1 else b for b in biases
        ]
        return new_weights, new_biases

class RandomNoise:
    def __init__(self,noise_min,noise_max):
        self.noise_min = noise_min
        self.noise_max = noise_max
    def __call__(self, weights, biases):
        noise_scale = ((self.noise_max - self.noise_min) * torch.rand(1) + self.noise_min).item()
        new_weights = [w + w.std() * noise_scale for w in weights]
        new_biases = [
            b + b.std() * noise_scale if b.shape[0] > 1 else b for b in biases
        ]
        return new_weights, new_biases

class Rotate:
    def __init__(self, rotation_degree):
        super().__init__()
        self.rotation_degree = rotation_degree
        angle = torch.empty(1).uniform_(-rotation_degree, rotation_degree)
        angle_rad = angle * (torch.pi / 180)
        self.rotation_matrix = torch.tensor(
            [
                [torch.cos(angle_rad), -torch.sin(angle_rad)],
                [torch.sin(angle_rad), torch.cos(angle_rad)],
            ]
        )

    def __call__(self, weights, biases):
        weights[0] = self.rotation_matrix @ weights[0]
        return weights, biases

class RandomRotate:
    def __init__(self, min_deg, max_deg):
        self.min_deg = min_deg
        self.max_deg = max_deg
    def __call__(self, weights, biases):

        rotation_degree = torch.randint(low=self.min_deg, high=self.max_deg, size=(1,)).item()
        angle = torch.empty(1).uniform_(-rotation_degree, rotation_degree)
        angle_rad = angle * (torch.pi / 180)
        self.rotation_matrix = torch.tensor(
            [
                [torch.cos(angle_rad), -torch.sin(angle_rad)],
                [torch.sin(angle_rad), torch.cos(angle_rad)],
            ]
        )
        weights[0] = self.rotation_matrix @ weights[0]
        return weights, biases

class Dropout:
    def __init__(self, p):
        self.drop_rate = p

    def __call__(self, weights, biases):
        new_weights = [F.dropout(w, p=self.drop_rate) for w in weights]
        new_biases = [F.dropout(w, p=self.drop_rate) for w in biases]
        return new_weights, new_biases

class Scale:
    def __init__(self, resize_scale):
        self.resize_scale = resize_scale
    def __call__(self, weights, biases):
        rand_scale = 1 + (torch.rand(1).item() - 0.5) * 2 * self.resize_scale
        weights[0] = weights[0] * rand_scale
        return weights, biases

class RandomScale:
    def __init__(self, min_scale, max_scale):
        self.min_scale = min_scale
        self.max_scale = max_scale
        # self.transform = Scale(scale.item())
    def __call__(self, weights, biases):
        resize_scale = ((self.max_scale - self.min_scale) * torch.rand(1) + self.min_scale).item()
        rand_scale = 1 + (torch.rand(1).item() - 0.5) * 2 * resize_scale
        weights[0] = weights[0] * rand_scale
        return weights, biases

class PositiveScale:
    def __init__(self, pos_scale):
        self.pos_scale = pos_scale
    def __call__(self, weights, biases):
        for i in range(len(weights) - 1):
            # todo: we do a lot of duplicated stuff here
            out_dim = biases[i].shape[0]
            scale = torch.from_numpy(
                np.random.uniform(
                    1 - self.pos_scale, 1 + self.pos_scale, out_dim
                ).astype(np.float32)
            )
            inv_scale = 1.0 / scale
            weights[i] = weights[i] * scale
            biases[i] = biases[i] * scale
            weights[i + 1] = (weights[i + 1].T * inv_scale).T
        return weights, biases

class QuantileDropout:
    def __init__(self, quantile_d):
        self.quantile_dropout = quantile_d
    def __call__(self, weights, biases):
        do_q = torch.empty(1).uniform_(0, self.quantile_dropout)
        q = torch.quantile(
            torch.cat([v.flatten().abs() for v in weights + biases]), q=do_q
        )
        new_weights = [torch.where(w.abs() < q, 0, w) for w in weights]
        new_biases = [torch.where(w.abs() < q, 0, w) for w in biases]
        return new_weights, new_biases
# class RandomCrop:
#     def __init__(self, width, exclude_missing_threshold=None):
#         self.width = width
#         self.exclude_missing_threshold = exclude_missing_threshold
#         assert exclude_missing_threshold is None or 0 <= exclude_missing_threshold <= 1
#
#     def __call__(self, x, mask=None, info=None, step=None):
#         if isinstance(x, np.ndarray):
#             if len(x.shape) == 1:
#                 x = x[:, np.newaxis]
#             if 'left_crop' in info:
#                 left_crop = info['left_crop']
#             else:
#                 seq_len = x.shape[0]
#                 if seq_len <= self.width:
#                     left_crop = 0
#                     warnings.warn(
#                         'cannot crop because width smaller than sequence length')
#                 else:
#                     left_crop = np.random.randint(seq_len - self.width)
#                 info['left_crop'] = left_crop
#                 info['right_crop'] = left_crop + self.width
#             out_x = x[left_crop:left_crop + self.width]
#             if mask is None:
#                 return (out_x, mask, info)
#             if self.exclude_missing_threshold is not None and np.isnan(out_x).mean() >= self.exclude_missing_threshold:
#                 return self.__call__(x, mask=mask, info=info)
#             out_m = mask[left_crop:left_crop + self.width]
#
#             return (out_x, out_m, info)
#         else:
#             raise NotImplementedError

class Permute():
    def __init__(self):
        pass
    def __call__(self,weights,biases):
        new_weights = [None] * len(weights)
        new_biases = [None] * len(biases)
        assert len(weights) == len(biases)

        perms = []
        for i, w in enumerate(weights):
            if i != len(weights) - 1:
                perms.append(torch.randperm(w.shape[1]))

        for i, (w, b) in enumerate(zip(weights, biases)):
            if i == 0:
                new_weights[i] = w[:, perms[i]]
                new_biases[i] = b[perms[i]]
            elif i == len(weights) - 1:
                new_weights[i] = w[perms[-1], :]
                new_biases[i] = b
            else:
                new_weights[i] = w[perms[i - 1], :][:, perms[i]]
                new_biases[i] = b[perms[i]]
        return new_weights, new_biases


    def __repr__(self):
        return "Permute"

class Binarize():
    def __init__(self, frac=0.1):
        self.frac = frac

    def __call__(self, weights, biases):
        new_weights = [None] * len(weights)
        new_biases = [None] * len(biases)
        assert len(weights) == len(biases)
        for i, (w, b) in enumerate(zip(weights, biases)):
            new_weights[i] = w
            new_biases[i] = b
            if i == 0:
                indices = torch.randperm(int(w.shape[1]*self.frac))
                sample_w = w[:, indices]
                sample_b = b[indices]
                new_weights[i][:, indices] = torch.where(sample_w < 0, -torch.ones_like(sample_w).float(),
                                                         torch.ones_like(sample_w).float())
                new_biases[i][indices] = torch.where(sample_b < 0, -torch.ones_like(sample_b).float(),
                                            torch.ones_like(sample_b).float())
        return new_weights, new_biases




class Identity():
    def __init__(self):
        pass

    def __call__(self, weights, biases):
        return weights, biases

    def __repr__(self):
        return "Identity"


class RandomTransform():
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, x, mask=None, info=None, step=None):
        t = np.random.choice(self.transforms, p=self.p)
        if 'random_transform' in info:
            info['random_transform'].append(str(t))
        else:
            info['random_transform'] = [str(t)]
        x, mask, info = t(x, mask=mask, info=info, step=step)
        return x, mask, info

    def __repr__(self):
        return f"RandomTransform(p={self.p})"