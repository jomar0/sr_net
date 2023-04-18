import torch
import copy
import os


def __extract_y__(image):
    y, _, _ = image.convert('YCbCr').split()
    return y


def psnr(y_true, y_pred):
    """
    Calculates the peak signal-to-noise ratio (PSNR) for two image tensors representing
    the Y channel of a YCbCr image.

    Args:
        y_true (torch.Tensor): The ground truth Y channel image tensor.
        y_pred (torch.Tensor): The predicted Y channel image tensor.

    Returns:
        float: The PSNR value in decibels.
    """
    mse = torch.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return float('inf')
    else:
        max_value = 1.0  # The maximum pixel value in the Y channel is 1.0
        psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
        return psnr.item()


class BestModel:
    def __init__(self, path=None):
        self.best_epoch = None
        self.best_model = None
        self.current_model = None
        self.best_psnr = -float('inf')

        if path is not None:
            self.load(path)

    def update(self, epoch, model, psnr):
        self.current_model = copy.deepcopy(model.state_dict())
        if psnr > self.best_psnr:
            self.best_epoch = epoch
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_psnr = psnr

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
                "best_psnr": self.best_psnr
            }, path)
        except OSError:
            default_filename = os.path.join(
                '/home/s.1909943/torch/', os.path.basename(path))
            torch.save(path.state_dict(), default_filename)

        def load(self, path):
            state = torch.load(path)
                
            self.best_epoch = state['best_epoch']
            self.best_model = state['best_model_state_dict']
            self.current_model = state['last_model_state_dict']
            self.best_psnr = state['best_psnr']
