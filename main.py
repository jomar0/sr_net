from data import ImagePairs
from train import *
from shrinknet import *
from evnet import *
from resblocknet import *
import os
from torch import nn
import time
from util import *
import argparse
import json
import shutil

parser = argparse.ArgumentParser(description="Trainer for SRNETs")
parser.add_argument("path", type=str, help="Program Argment json path")
cmd_args = parser.parse_args()
args = read_json_file(cmd_args.path)

training, evaluation, testing = create_datasets(args["dataset"])
log_path = args["name"] + ".log"
model_path = args["name"] + ".model"
log_path = os.path.join(cmd_args.path, log_path)
model_path = os.path.join(cmd_args.path, model_path)

model = create_model(args)
loss = create_loss(args)

# file.write the arguments
open(log_path, "w").close()  # delete content
with open(log_path, "a") as file:
    file.write("=" * 80 + "\n")
    file.write(f"Log File for Training Session\n")

    file.write("=" * 80 + "\n")

print("=" * 80)
print(f"{json.dumps(args, indent=True)}\n")
print("=" * 80)


start = time.time()
results = train(
    model=model,
    dataloaders=create_dataloaders(
        training, evaluation, num_workers=0, batch_size=args["batch_size"]
    ),
    epochs=args["epochs"],
    learning_rate=args["learning_rate"],
    log_path=log_path,
    criterion=loss
)

elapsed = time.time() - start
formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed))

with open(log_path, "a") as file:
    file.write(f"Training took: {formatted_time}" + "\n")


sample_path = create_sample_folder(os.path.join(cmd_args.path, "samples"))
same_epoch = results.data["best_ssim"]["epoch"] != results.data["best_psnr"]["epoch"]
for metric in (["ssim", "psnr"] if same_epoch else ["ssim"]):
    model.load_state_dict(results.data[f"best_{metric}"]["model"])
    val_ssim, val_psnr, val_loss = test(
        model=model, dataloader=DataLoader(testing, batch_size=1), criterion=create_loss(args), device="cuda")
    results.update_test(metric if same_epoch else "both", val_psnr, val_ssim, val_loss, loss=loss.__class__.__name__)

    
    #generate the samples
    try:
        sample_ids = args["sample_ids"]
        if not isinstance(sample_ids, list) or not all(isinstance(i, int) for i in sample_ids): raise KeyError("sample_ids is not a list of ints")
    except KeyError:
        print("No Samples Specified, using default samples")
        sample_ids = [16,8,30]
    for i in sample_ids:
        sr, hr = generate_sample(i, testing, model)
        name_base = str(i).zfill(3)
        sr.save(os.path.join(sample_path, name_base + "_SR.png" ))
        hr.save(os.path.join(sample_path, name_base + "_HR.png" ))

# ssim params
model.load_state_dict(results.data["best_ssim"]["model"])
params = sum(p.numel() for p in model.parameters())
results.data["best_ssim"]["model_params"] = params
with open(log_path, "a") as file:
    file.write(f"Best SSIM - Number of Parameters: {params}\n")
print(f"Best SSIM - Number of Parameters: {params}")

# psnr params
model.load_state_dict(results.data["best_psnr"]["model"])
params = sum(p.numel() for p in model.parameters())
results.data["best_psnr"]["model_params"] = params
with open(log_path, "a") as file:
    file.write(f"Best PSNR - Number of Parameters: {params}\n")
print(f"Best PSNR - Number of Parameters: {params}")


results.save(model_path)
print("Complete")