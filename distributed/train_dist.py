import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from utils import psnr, BestModel


def init_dist_env():
    dist_url = "env://"
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )

    torch.cuda.set_device(local_rank)

    dist.barrier()


def create_dataloaders(training_dataset, evaluation_dataset, batch_size=16, num_workers=5):
    training_sampler = DistributedSampler(
        dataset=training_dataset, shuffle=True)
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size,
                                     sampler=training_sampler, num_workers=num_workers, pin_memory=True)
    evaluation_dataloader = DataLoader(
        dataset=evaluation_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return training_dataloader, evaluation_dataloader


def train(model, dataloaders, epochs, save_path, learning_rate):
    best = BestModel()
    # Setup optim & critera
    criterion = nn.MSELoss()
    optimiser = optim.Adam(params=model.parameters(), lr=learning_rate)

    # Untuple dataloaders
    training_dataloader, evaluation_dataloader = dataloaders

    for epoch in range(epochs):
        training_dataloader.sampler.set_epoch(epoch)  # new epoch time

        # Train the Model
        model.train()
        for inputs, ground_truths in training_dataloader:
            inputs = inputs.cuda()  # pop them on the GPU
            ground_truths = ground_truths.cuda()

            optimiser.zero_grad()

            outputs = model.inputs()
            loss = criterion(outputs, ground_truths)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * inputs.size(0)  # update loss
        epoch_loss /= len(training_dataloader.dataset)  # calc loss || |_

        # Report time
        model.eval()
        eval_psnr = 0.0
        with torch.no_grad():
            for input, ground_truth in evaluation_dataloader:
                input = input.cuda()  # pop them on the GPU, again
                ground_truth = ground_truth.cuda()

                output = model(input)

                eval_psnr += psnr(output, ground_truth)
            eval_psnr / + len(evaluation_dataloader.dataset)
            best.update(epoch=epoch, model=model, psnr=eval_psnr)
        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, PSNR: {eval_psnr:.2f} dB")
    return best
