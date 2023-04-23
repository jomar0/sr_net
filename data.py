from torch.utils.data import Dataset
from torchvision import transforms
from util import extract_y
from PIL import Image
from pathlib import Path

class ImagePairs(Dataset):
    def __init__(self, dataset_directory):
        super(ImagePairs, self).__init__()
        self.directory = Path(dataset_directory)
        self.count = len(list((self.directory / 'HR').glob('*.png')))
        self.transforms = transforms.Compose([
            transforms.Lambda(extract_y),
            transforms.ToTensor()
        ])

    def transform(self, input):
        return self.transforms(input)
    def __getitem__(self, index):
        name_base = str(index).zfill(5)
        name_lr = name_base + '_LR.png'
        name_hr = name_base + '_HR.png'

        lr_image = Image.open(self.directory / 'LR' / name_lr)

        hr_image = Image.open(self.directory / 'HR' / name_hr)

        lr_image = self.transforms(lr_image)
        hr_image = self.transforms(hr_image)

        return (lr_image, hr_image)
    
    def get_item(self, index):
        name_base = str(index).zfill(5)
        name_lr = name_base + "_LR.png"
        name_hr = name_base + "_HR.png"
        lr_image = Image.open(self.directory / 'LR' / name_lr)
        hr_image = Image.open(self.directory / 'HR' / name_hr)
        return lr_image, hr_image
    
    def __len__(self):
        return self.count