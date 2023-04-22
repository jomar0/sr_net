import torch
import torch.nn as nn
import copy
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import torch
from skimage.metrics import structural_similarity

def initialise(self, module):
    nn.init.kaiming_normal_(module.weight, mode="fan_out")
    nn.init.constant_(module.bias)
    return module

def create_dataloaders(training_dataset, evaluation_dataset, batch_size=16, num_workers=5):
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    evaluation_dataloader = DataLoader(
        dataset=evaluation_dataset, batch_size=1, num_workers=num_workers, pin_memory=True)

    return training_dataloader, evaluation_dataloader

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

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1

    def create_window(self):
        # Create a 1D Gaussian window
        window = torch.Tensor([np.exp(-(x - self.window_size//2)**2/float(2*2)) for x in range(self.window_size)])
        # Make it 2D
        window_2d = window.outer(window)
        # Make it 3D with one channel
        window_3d = window_2d.unsqueeze(0).unsqueeze(0)
        # Repeat the 3D window for all channels
        window_3d = window_3d.repeat(self.channel, 1, 1, 1)
        return window_3d

    def forward(self, img1, img2):
        if len(img1.shape) != 4 or len(img2.shape) != 4:
            raise ValueError("Input tensors must have a batch dimension")

        if img1.shape != img2.shape:
            raise ValueError("Input tensors must have the same shape")

        if img1.shape[1] != 1:
            raise ValueError("Input tensors must be grayscale")

        window = self.create_window()

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=self.channel) - mu1_mu2

        c1 = (0.01 * 255)**2
        c2 = (0.03 * 255)**2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        if self.size_average:
            return torch.mean((1 - ssim_map) / 2)
        else:
            return torch.sum((1 - ssim_map) / 2)


class Model:
    def __init__(self):
        self.data = dict()
    def update_eval(self, epoch, model, ssim, psnr):
        self.data["current_model"] = copy.deepcopy(model.state_dict())
        self.data["model_args"] = copy.deepcopy(model.args)
        for metric in ["ssim", "psnr"]:
            try:
                if self.data[f"best_{metric}"]["eval"]["ssim"] > locals()[metric]:
                    self.data[f"best_{metric}"]["epoch"] = epoch
                    self.data[f"best_{metric}"]["eval"]["ssim"] = ssim
                    self.data[f"best_{metric}"]["eval"]["psnr"] = psnr
                    self.data[f"best_{metric}"]["model"] = copy.deepcopy(model.state_dict())
            except ValueError:
                self.data[f"best_{metric}"]["epoch"] = epoch
                self.data[f"best_{metric}"]["eval"]["ssim"] = ssim
                self.data[f"best_{metric}"]["eval"]["psnr"] = psnr
                self.data[f"best_{metric}"]["model"] = copy.deepcopy(model.state_dict())
    def update_test(self, value_type, psnr, ssim, loss, loss_type):
        for metric in (["ssim", "psnr"] if value_type == "both" else [value_type]):
            self.data[f"best_{metric}"]["test"]["ssim"] == ssim
            self.data[f"best_{metric}"]["test"]["psnr"] == psnr
            self.data[f"best_{metric}"]["test"]["loss"]["value"] == loss
            self.data[f"best_{metric}"]["test"]["loss"]["type"] == loss_type
    def save(self, path):
        torch.save(self.data, path)
    def load(self, path):
        torch.load(path)