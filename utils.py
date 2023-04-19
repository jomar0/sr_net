import torch
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

class BestModel:
    def __init__(self, path=None):
        self.current_model = None

        self.best_epoch_psnr = None
        self.best_model_psnr = None
        self.best_psnr = -float('inf')

        self.best_epoch_ssim = None
        self.best_model_ssim = None
        self.best_ssim = 0.0

        if path is not None:
            self.load(path)

    def update(self, epoch, model, ssim, psnr):
        self.current_model = copy.deepcopy(model.state_dict())
        if ssim > self.best_ssim:
            self.best_epoch_ssim = epoch
            self.best_model_ssim = copy.deepcopy(model.state_dict())
            self.best_ssim = ssim
        if psnr > self.best_psnr:
            self.best_epoch_psnr = epoch
            self.best_model_psnr = copy.deepcopy(model.state_dict())
            self.best_psnr = psnr

    def load_best_model_psnr(self, model):
        model.load_state_dict(self.best_model_psnr)

    def load_best_model_ssim(self, model):
        model.load_state_dict(self.best_model_ssim)

    def load_last_model(self, model):
        model.load_state_dict(self.current_model)

    def save(self, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                "best_model_psnr_state_dict": self.best_model_psnr,
                "best_epoch_psnr": self.best_epoch_psnr,
                "best_psnr": self.best_psnr,

                "best_model_ssim_state_dict": self.best_model_ssim,
                "best_epoch_ssim": self.best_epoch_ssim,
                "best_ssim": self.best_ssim,

                "last_model_state_dict": self.current_model,
            }, path)
        except OSError:
            default_filename = os.path.join(
                '~/torch/', os.path.basename(path))
            torch.save(path.state_dict(), default_filename)

        def load(self, path):
            state = torch.load(path)
                
            self.best_epoch_psnr = state['best_epoch_psnr']
            self.best_model_psnr = state['best_model_psnr_state_dict']
            self.best_psnr = state['best_psnr']

            self.best_epoch_ssim = state['best_epoch_ssim']
            self.best_model_ssim = state['best_model_ssim_state_dict']
            self.best_ssim = state['best_ssim']

            self.current_model = state['last_model_state_dict']


class BestModel_old:
    def __init__(self, path=None):
        self.best_epoch = None
        self.best_model = None
        self.current_model = None
        self.best_value = -float('inf')

        if path is not None:
            self.load(path)

    def update(self, epoch, model, value):
        self.current_model = copy.deepcopy(model.state_dict())
        if value > self.best_value:
            self.best_epoch = epoch
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_value = value

    def load_best_model(self, model):
        model.load_state_dict(self.best_model)

    def load_last_model(self, model):
        model.load_state_dict(self.current_model)

    def save(self, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                "best_model_state_dict": self.best_model,
                "last_model_state_dict": self.current_model,
                "best_epoch": self.best_epoch,
                "best_value": self.best_value,
            }, path)
        except OSError:
            default_filename = os.path.join(
                '~/torch/', os.path.basename(path))
            torch.save(path.state_dict(), default_filename)

        def load(self, path):
            state = torch.load(path)
                
            self.best_epoch = state['best_epoch']
            self.best_model = state['best_model_state_dict']
            self.current_model = state['last_model_state_dict']
            self.best_value = state['best_value']