import torch
import torch.nn as nn
import copy
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import torch
from skimage.metrics import structural_similarity

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
    def update(self, epoch, model, ssim, psnr):

        self.data["psnr"]["value"] = psnr
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
    
            



# class BestModel_old2:
#     def __init__(self, path=None):
#         self.current_model = None

#         self.best_epoch_psnr = None
#         self.best_model_psnr = None
#         self.best_psnr = -float('inf')

#         self.best_epoch_ssim = None
#         self.best_model_ssim = None
#         self.best_ssim = 0.0

#         if path is not None:
#             self.load(path)

#     def update(self, epoch, model, ssim, psnr):
#         self.current_model = copy.deepcopy(model.state_dict())
#         if ssim > self.best_ssim:
#             self.best_epoch_ssim = epoch
#             self.best_model_ssim = copy.deepcopy(model.state_dict())
#             self.best_ssim = ssim
#         if psnr > self.best_psnr:
#             self.best_epoch_psnr = epoch
#             self.best_model_psnr = copy.deepcopy(model.state_dict())
#             self.best_psnr = psnr

#     def load_best_model_psnr(self, model):
#         model.load_state_dict(self.best_model_psnr)

#     def load_best_model_ssim(self, model):
#         model.load_state_dict(self.best_model_ssim)

#     def load_last_model(self, model):
#         model.load_state_dict(self.current_model)

#     def save(self, path):
#         try:
#             os.makedirs(os.path.dirname(path), exist_ok=True)
#             torch.save({
#                 "best_model_psnr_state_dict": self.best_model_psnr,
#                 "best_epoch_psnr": self.best_epoch_psnr,
#                 "best_psnr": self.best_psnr,

#                 "best_model_ssim_state_dict": self.best_model_ssim,
#                 "best_epoch_ssim": self.best_epoch_ssim,
#                 f"best_{metric}": self.best_ssim,

#                 "last_model_state_dict": self.current_model,
#             }, path)
#         except OSError:
#             default_filename = os.path.join(
#                 '~/torch/', os.path.basename(path))
#             torch.save(path.state_dict(), default_filename)

#         def load(self, path):
#             state = torch.load(path)
                
#             self.best_epoch_psnr = state['best_epoch_psnr']
#             self.best_model_psnr = state['best_model_psnr_state_dict']
#             self.best_psnr = state['best_psnr']

#             self.best_epoch_ssim = state['best_epoch_ssim']
#             self.best_model_ssim = state['best_model_ssim_state_dict']
#             self.best_ssim = state['best_ssim']

#             self.current_model = state['last_model_state_dict']


# class BestModel_old:
#     def __init__(self, path=None):
#         self.best_epoch = None
#         self.best_model = None
#         self.current_model = None
#         self.best_value = -float('inf')

#         if path is not None:
#             self.load(path)

#     def update(self, epoch, model, value):
#         self.current_model = copy.deepcopy(model.state_dict())
#         if value > self.best_value:
#             self.best_epoch = epoch
#             self.best_model = copy.deepcopy(model.state_dict())
#             self.best_value = value

#     def load_best_model(self, model):
#         model.load_state_dict(self.best_model)

#     def load_last_model(self, model):
#         model.load_state_dict(self.current_model)

#     def save(self, path):
#         try:
#             os.makedirs(os.path.dirname(path), exist_ok=True)
#             torch.save({
#                 "best_model_state_dict": self.best_model,
#                 "last_model_state_dict": self.current_model,
#                 "best_epoch": self.best_epoch,
#                 "best_value": self.best_value,
#             }, path)
#         except OSError:
#             default_filename = os.path.join(
#                 '~/torch/', os.path.basename(path))
#             torch.save(path.state_dict(), default_filename)

#         def load(self, path):
#             state = torch.load(path)
                
#             self.best_epoch = state['best_epoch']
#             self.best_model = state['best_model_state_dict']
#             self.current_model = state['last_model_state_dict']
#             self.best_value = state['best_value']