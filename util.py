import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity
import math

def generate_sample(id, dataset, model, device="cuda"):
    to_PIL = transforms.ToPILImage()
    lr, hr = dataset.get_item(id)
    input = dataset.transform(lr)
    interpolated = lr.resize((1280, 720), Image.Resampling.BICUBIC)
    _, cb, cr = interpolated.convert("YCbCr").split()

    input = input.to(device)
    sr_y = model(input).clamp(0.0,1.0)
    sr_y = sr_y.cpu().detach()
    return Image.merge("YCbCr", (to_PIL(sr_y), cb, cr)), hr
    

def initialise(module):
    nn.init.kaiming_normal_(module.weight, mode="fan_out")
    nn.init.constant_(module.bias, 0.01)
    return module

def extract_y(image):
    y, _, _ = image.convert('YCbCr').split()
    return y

def ssim(y_pred, y_true):
    # Convert tensors to numpy arrays
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    # Compute SSIM
    ssim_value = structural_similarity(y_true[0, 0], y_pred[0, 0], data_range=y_true.max() - y_true.min())

    return ssim_value

def psnr(y_pred, y_true):
    # Convert tensors to numpy arrays
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    # Calculate MSE
    mse = np.mean((y_true - y_pred) ** 2)

    # Calculate maximum pixel value

    # Calculate value in dB
    value = 20 * np.log10(1.0) - 10 * np.log10(mse)
    return value

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, sigma=1.5):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.sigma = sigma
        self.window = self.create_window(window_size, sigma)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, sigma):
        _1D_window = self.gaussian(window_size, sigma)
        _2D_window = _1D_window.outer(_1D_window)
        return _2D_window.view(1, 1, window_size, window_size).to(torch.device("cuda"))

    def forward(self, x, y):
        mu1 = F.conv2d(x, self.window, padding=self.window_size//2, groups=x.shape[1])
        mu2 = F.conv2d(y, self.window, padding=self.window_size//2, groups=y.shape[1])

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(x * x, self.window, padding=self.window_size//2, groups=x.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(y * y, self.window, padding=self.window_size//2, groups=y.shape[1]) - mu2_sq
        sigma12 = F.conv2d(x * y, self.window, padding=self.window_size//2, groups=x.shape[1]) - mu1_mu2

        C1 = (0.01)**2
        C2 = (0.03)**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return torch.mean((1 - ssim_map) / 2)
        else:
            return torch.mean(torch.mean((1 - ssim_map) / 2, dim=(2,3)), dim=1)