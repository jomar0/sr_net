import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import psnr, Model, ssim, SSIMLoss
import statistics
import copy
import os


def train(model, dataloaders, epochs, learning_rate, criterion, log_path=None, device='cuda'):
    save_state = Model()
    optimiser = optim.Adam(params=model.parameters(), lr=learning_rate)

    training_dataloader, evaluation_dataloader = dataloaders

    device = torch.device(device)
    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        progress_bar = tqdm(desc=f'Epoch {epoch+1}/{epochs} - Training  ', total=len(
            training_dataloader.dataset), postfix=f'Loss: 0.000')
        for data in training_dataloader:
            inputs, ground_truths = data
            inputs = inputs.to(device)  # pop them on the GPU
            ground_truths = ground_truths.to(device)

            optimiser.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, ground_truths)
            loss.backward()
            optimiser.step()
            progress_bar.update(len(inputs))
            epoch_loss += loss.item() * inputs.size(0)  # update loss
            loss_so_far = epoch_loss / progress_bar.n
            progress_bar.set_postfix_str(f'Loss: {loss_so_far:.6f}')
        epoch_loss /= len(training_dataloader.dataset)  # calc loss || |_
        progress_bar.set_description_str(
            f"Epoch {epoch+1}/{epochs} - Evaluating")
        progress_bar.total = len(evaluation_dataloader.dataset)
        progress_bar.n = 0
        model.eval()
        eval_psnr = 0.0
        eval_ssim = 0.0
        with torch.no_grad():
            for input, ground_truth in evaluation_dataloader:
                input = input.to(device)  # pop them on the GPU, again
                ground_truth = ground_truth.to(device)

                output = model(input)

                eval_psnr += psnr(output, ground_truth)
                eval_ssim += ssim(output, ground_truth)
                progress_bar.update(len(input))
            eval_psnr /= len(evaluation_dataloader.dataset)
            eval_ssim /= len(evaluation_dataloader.dataset)
            save_state.update(epoch=epoch+1, model=model,
                              psnr=eval_psnr, ssim=eval_ssim)
        progress_bar.close()
        status = f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}, PSNR: {eval_psnr:.6f} dB, SSIM: {eval_ssim:.6f}"
        print(status)
        if log_path is not None:
            if os.path.exists(log_path):
                with open(log_path, "a") as file:
                    file.write(status + "\n")
    return save_state


def test(model, dataloader, criterion=None, log_path=None, device='cuda'):
    if criterion == None:
        criterion = SSIMLoss()
    model.to(device)
    model.eval()
    eval_psnr = 0.0
    eval_ssim = 0.0
    eval_loss = 0.0
    progress_bar = tqdm(desc="Testing", total=len(dataloader.dataset))
    with torch.no_grad():
        for inputs, ground_truths in dataloader:
            inputs = inputs.to(device)
            ground_truths = ground_truths.to(device)

            outputs = model(inputs).clamp(0.0, 1.0)
            loss = criterion(outputs, ground_truths)
            eval_loss += loss.item()
            eval_psnr += psnr(outputs, ground_truths)
            eval_ssim += psnr(outputs, ground_truths)
            progress_bar.update(len(inputs))
        eval_psnr /= len(dataloader.dataset)
        eval_ssim /= len(dataloader.dataset)
        eval_loss /= len(dataloader.dataset)

        
        progress_bar.close()
        status = f"Loss: {eval_loss:.6f}, PSNR: {eval_psnr:.6f} dB, SSIM: {eval_ssim:.6f}"
        print(status)
        if log_path is not None:
            if os.path.exists(log_path):
                with open(log_path, "a") as file:
                    file.write(status + "\n")
