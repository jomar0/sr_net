import torch
from data import ImagePairs
from utils import create_dataloaders, psnr, ssim
training_dataset = ImagePairs("/home/u1909943/MSc/ImagePairs/training")
device = torch.device("cuda")
evaluation_dataset = ImagePairs("/home/u1909943/MSc/ImagePairs/evaluation")
_, dataloader = create_dataloaders(training_dataset, evaluation_dataset, num_workers=0, batch_size=16)
avg_psnr = 0.0
avg_ssim = 0.0
for data in dataloader:
    input, gt = data
    input.to(device)
    gt.to(device)

    output = torch.nn.functional.interpolate(input, scale_factor=2, mode='bicubic')
    avg_psnr += psnr(output, gt)
    avg_ssim += ssim(output, gt)
avg_psnr = avg_psnr / len(dataloader.dataset)
avg_ssim = avg_ssim / len(dataloader.dataset)

print(f"PSNR: {avg_psnr:.6f}")
print(f"SSIM: {avg_ssim:.6f}")