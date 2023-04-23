import copy
from torch import save, load
class Model:
    def __init__(self):
        self.data = dict()
        self.data["best_ssim"] = dict()
        self.data["best_psnr"] = dict()
        self.data["best_ssim"]["eval"] = dict()
        self.data["best_psnr"]["eval"] = dict()

        self.data["best_ssim"]["test"] = dict()
        self.data["best_ssim"]["test"]["loss"] = dict()
        self.data["best_psnr"]["test"] = dict()
        self.data["best_psnr"]["test"]["loss"] = dict()


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
            except KeyError:
                self.data[f"best_{metric}"]["epoch"] = epoch
                self.data[f"best_{metric}"]["eval"]["ssim"] = ssim
                self.data[f"best_{metric}"]["eval"]["psnr"] = psnr
                self.data[f"best_{metric}"]["model"] = copy.deepcopy(model.state_dict())
    def update_test(self, value_type, psnr, ssim, loss, loss_type):
        for metric in (["ssim", "psnr"] if value_type == "both" else [value_type]):
            self.data[f"best_{metric}"]["test"]["ssim"] = ssim
            self.data[f"best_{metric}"]["test"]["psnr"] = psnr
            self.data[f"best_{metric}"]["test"]["loss"]["value"] = loss
            self.data[f"best_{metric}"]["test"]["loss"]["type"] = loss_type
    def save(self, path):
        save(self.data, path)
    def load(self, path):
        load(path)