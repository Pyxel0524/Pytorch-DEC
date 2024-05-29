# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:11:10 2024

ICRA 2024

@author: zpy
"""
import torch
from torch import nn
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pywt
import torch.fft as fft


class DEC(nn.Module):
    def __init__(self, d0):
        super().__init__()
        self.low_pass_d = d0

    def low_pass_filter(self, img, d):
        """
        Low-pass filter using a circular mask in the frequency domain.

        Args:
        - image_fft: Fourier transformed image
        - d: Cut-off frequency

        Returns:
        - Filtered Fourier transformed image
        """
        # Create a circular mask
        image_fft = fft.fftn(img, dim=(-2, -1))

        h, w = image_fft.shape[-2:]
        cy, cx = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= d ** 2
        mask = mask.float()

        # Apply the mask to the image in the frequency domain
        filtered_image_fft = image_fft * mask
        filtered_image = fft.ifftn(filtered_image_fft, dim=(-2, -1)).real

        return filtered_image

    def binary_process(self, a):
        deep = a.size(3)
        a_s = a[:, :, :, :int(deep / 2), :]
        a_d = a[:, :, :, int(deep / 2):, :]
        thre_s = torch.sum(a_s, 3) / a_s.size(3)
        thre_d = torch.sum(a_d, 3) / a_d.size(3)
        for i in range(a.size(4)):
            a_s[:, :, :, :, i][a_s[:, :, :, :, i] > thre_s[:, :, :, i].unsqueeze(3)] = 255
            a_s[:, :, :, :, i][a_s[:, :, :, :, i] <= thre_s[:, :, :, i].unsqueeze(3)] = 0
            a_d[:, :, :, :, i][a_d[:, :, :, :, i] > thre_d[:, :, :, i].unsqueeze(3)] = 255
            a_d[:, :, :, :, i][a_d[:, :, :, :, i] <= thre_d[:, :, :, i].unsqueeze(3)] = 0
        return torch.tensor(np.concatenate([a_s, a_d], axis=3))

    def energy_center(self, b):
        batch = b.size(0)
        deep = b.size(3)
        trace = b.size(4)
        b_s = b[:, :, :, :int(deep / 2), :]
        b_d = b[:, :, :, int(deep / 2):, :]
        p_s = np.zeros((batch, trace));
        s_p_s = np.zeros((batch, trace))
        p_d = np.zeros((batch, trace));
        s_p_d = np.zeros((batch, trace))
        for j in range(batch):
            for i in range(b.size(4)):
                q_s = torch.nonzero(b_s[j, :, :, :, i]).size(0)
                q_d = torch.nonzero(b_d[j, :, :, :, i]).size(0)
                p_s[j, i] = sum(torch.where(b_s[j, :, :, :, i] == 255)[2]) / q_s
                p_d[j, i] = sum(torch.where(b_d[j, :, :, :, i] == 255)[2]) / q_s + int(deep / 2)

            # 4. dwt
            coeffs = pywt.dwt(p_s[j, :], 'db1')
            cA, cD = coeffs
            s_p_s[j, :] = pywt.idwt(cA, None, 'db1')

            coeffs = pywt.dwt(p_d[j, :], 'db1')
            cA, cD = coeffs
            s_p_d[j, :] = pywt.idwt(cA, None, 'db1')

        return torch.tensor(np.concatenate([torch.tensor(s_p_s), torch.tensor(s_p_d)], axis=1))

    def forward(self, x):
        # 1. frequency filter
        x = self.low_pass_filter(x, self.low_pass_d)
        # 2. binary process
        x = self.binary_process(x)
        # 3. energy center
        x = self.energy_center(x)

        return x


if __name__ == "__main__":
    DEC = DEC(d0=30)
    # 读取图像
    img = torch.randn(4, 1, 1, 200, 202)
    dec_f = DEC(img)
    print(dec_f.shape)



