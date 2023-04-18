import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import psnr, BestModel
import statistics
import copy


    
def create_dataloaders(training_dataset, evaluation_dataset, batch_size=16, num_workers=5):
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    evaluation_dataloader = DataLoader(
        dataset=evaluation_dataset, batch_size=1, num_workers=num_workers, pin_memory=True)

    return training_dataloader, evaluation_dataloader

def train(model, dataloaders, epochs, learning_rate, device='cuda'):
    best = BestModel()
    criterion = nn.MSELoss()
    optimiser = optim.Adam(params=model.parameters(), lr=learning_rate)

    training_dataloader, evaluation_dataloader = dataloaders

    device = torch.device(device)
    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        progress_bar = tqdm(desc=f'Epoch {epoch}/{epochs}', total=len(training_dataloader.dataset), postfix=f'Loss: 0.000')
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
            progress_bar.set_postfix_str(f'Loss: {loss_so_far:.4f}')
        epoch_loss /= len(training_dataloader.dataset)  # calc loss || |_

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
        progress_bar.close()
        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, PSNR: {eval_psnr:.2f} dB")
    return best


def old_train(model, train_dataset, eval_dataset, epochs=10, batch_size=16, learning_rate=0.001, save_state=None, load_from_save=False, device='cuda'):
    
    # Setup Device
    device = torch.device(device)
    model.to(device)

    # Initialise Dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    eval_dataloader = DataLoader(
        dataset=eval_dataset, batch_size=1, pin_memory=True)

    # Initalise Training Criteria & Optimiser
    train_criterion = nn.MSELoss()
    train_optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    # Load from save
    if load_from_save and save_state is not None:
        state = torch.load(save_state)
        start_epoch = state["epoch"] + 1
        model.load_state_dict(state["model_state_dict"])
        train_optimiser.load_state_dict(state["optimiser_state_dict"])
        eval_results_store = state["eval_results_store"]
        best_weights = state["best_state_dict"]
    else:
        if (not load_from_save):
            print("Not loading from save state")
        if (save_state is None):
            print("Not saving progress!")
        start_epoch = 0
        eval_results_store = list([])
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    for epoch in range(start_epoch, epochs):
            # Initalise Epoch Metrics
            epoch_losses = list([])
            epoch_psnr_store = list([])
            
            progress_bar = tqdm(desc=f"Epoch {epoch}/{epochs}", total=len(train_dataset))
            # train an epoch
            model.train()
            for data in train_dataloader:
                input, ground_truth = data
                input = input.to(device)
                ground_truth = ground_truth.to(device)

                predictions = model(input)

                loss = train_criterion(predictions, ground_truth)

                train_optimiser.zero_grad()

                epoch_losses.append(loss.item())

                loss.backward()

                train_optimiser.step()
                progress_bar.update(16)
            print("Epoch\t{:03d}:\t{:.3f}".format(epoch, loss.item()), end=None)
            # evaluate an epoch
            model.eval()
            for data in eval_dataloader:
                input, ground_truth = data
                input = input.to(device)
                ground_truth = ground_truth.to(device)

                with torch.no_grad():
                    predictions = model(input).clamp(0.0, 1.0)
                epoch_psnr_store.append(psnr(predictions, ground_truth))

            # Update best weights if this is the best epoch
            if len(eval_results_store) == 0:
                best_weights = copy.deepcopy(model.state_dict())
            elif statistics.mean(epoch_psnr_store) > max(eval_results_store):
                best_weights = copy.deepcopy(model.state_dict())
            eval_results_store.append(statistics.mean(epoch_psnr_store))
            
            print("\tPSNR:{:.3f}dB".format(statistics.mean(epoch_psnr_store)))

            # Save the program state
            if save_state is not None:
                torch.save({
                    "epoch": epoch,
                    "eval_results_store": eval_results_store,
                    "best_state_dict": best_weights,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": train_optimiser.state_dict()
                }, save_state)
    # Print Completion Message
    print(
        "Training Complete\nThe trained model can be found at \"{save_state}\"\nStats:\n")
    print("\tBest PSNR:{}".format(max(eval_results_store)))
    print("\tBest Epoch:{}".format(
        eval_results_store.index(max(eval_results_store))))