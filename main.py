from data import ImagePairs
from train import *
from shrinking_based import *
from evnet import *
from nonshrinking_based import *
import os
from torch import nn
import time
from utils import Model, create_dataloaders, SSIMLoss
import argparse
import json
import shutil

def create_sample_folder(sample_path):
    if os.path.exists(sample_path):
    # If directory already exists, delete all files in it
        for filename in os.listdir(sample_path):
            file_path = os.path.join(sample_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
                return None
    else:
    # If directory does not exist, create it
        os.makedirs(sample_path)
    return sample_path

def read_json_file(file_path):
    file_path = os.path.join(file_path, "config.json")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found")
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def create_datasets(dataset_root_path):
    # Check if the directory exists
    if not os.path.isdir(dataset_root_path):
        raise NotADirectoryError(
            f"{dataset_root_path} is not a directory or does not exist"
        )

    # Check if the expected subdirectories exist and save their paths to variables
    training_path = os.path.join(dataset_root_path, "training")
    if not os.path.isdir(training_path):
        raise NotADirectoryError(
            f"{training_path} is not a directory or does not exist"
        )

    testing_path = os.path.join(dataset_root_path, "testing")
    if not os.path.isdir(testing_path):
        raise NotADirectoryError(
            f"{testing_path} is not a directory or does not exist")

    evaluation_path = os.path.join(dataset_root_path, "evaluation")
    if not os.path.isdir(evaluation_path):
        raise NotADirectoryError(
            f"{evaluation_path} is not a directory or does not exist"
        )

    # Check if the expected sub-subdirectories exist
    for subdir_path in [training_path, testing_path, evaluation_path]:
        subsubdirs = ["HR", "LR"]
        for subsubdir in subsubdirs:
            subsubdir_path = os.path.join(subdir_path, subsubdir)
            if not os.path.isdir(subsubdir_path):
                raise NotADirectoryError(
                    f"{subsubdir_path} is not a directory or does not exist"
                )
    return (
        ImagePairs(training_path),
        ImagePairs(evaluation_path),
        ImagePairs(testing_path),
    )


def create_log_directory(log_path, name):
    directory_path = os.path.join(log_path, name)

    # Create directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def create_model(args):
    # Parse model name
    model_name = args["model"]["name"]
    if ":" in model_name:
        superclass, subclass = model_name.split(":")
    else:
        superclass = model_name
        subclass = None

    # Get model configuration
    model_config = args["model"]["args"]

    # Validate model configuration
    if superclass == "ShrinkNet":
        required_params = [
            "feature_channels",
            "shrinking_channels",
            "mapping_depth",
            "kernels",
            "types",
        ]
    elif superclass == "EVNet":
        required_params = ["kernels", "channels"]
    elif superclass == "ResBlockNet":
        required_params = ["config"]
    else:
        raise ValueError(f"Invalid model superclass: {superclass}")

    for param in required_params:
        if param not in model_config:
            raise ValueError(
                f"Missing required parameter '{param}' for model '{model_name}'"
            )

    # Call appropriate subfunction to create model object instance
    if superclass == "ShrinkNet":
        if subclass == "ShrinkNet_Residual1":
            return ShrinkNet_Residual1(
                feature_channels=model_config["feature_channels"],
                shrinking_channels=model_config["shrinking_channels"],
                mapping_depth=model_config["mapping_depth"],
                types = model_config["types"],
                kernels=[tuple(x) for x in model_config["kernels"]],
            )
        elif subclass == "ShrinkNet_Residual2":
            return ShrinkNet_Residual2(
                feature_channels=model_config["feature_channels"],
                shrinking_channels=model_config["shrinking_channels"],
                mapping_depth=model_config["mapping_depth"],
                types = model_config["types"],
                kernels=[tuple(x) for x in model_config["kernels"]],
            )
        elif subclass == "ShrinkNet_Residual3":
            return ShrinkNet_Residual3(
                feature_channels=model_config["feature_channels"],
                shrinking_channels=model_config["shrinking_channels"],
                mapping_depth=model_config["mapping_depth"],
                types = model_config["types"],
                kernels=[tuple(x) for x in model_config["kernels"]],
            )
        elif subclass == "ShrinkNet_Residual4":
            return ShrinkNet_Residual4(
                feature_channels=model_config["feature_channels"],
                shrinking_channels=model_config["shrinking_channels"],
                mapping_depth=model_config["mapping_depth"],
                types = model_config["types"],
                kernels=[tuple(x) for x in model_config["kernels"]],
            )
        else:
            return ShrinkNet(
                feature_channels=model_config["feature_channels"],
                shrinking_channels=model_config["shrinking_channels"],
                mapping_depth=model_config["mapping_depth"],
                types = model_config["types"],
                kernels=[tuple(x) for x in model_config["kernels"]],
            )
    elif superclass == "EVNet":
        return EVNet(kernels=[tuple(x) for x in model_config["kernels"]], channels=model_config["channels"])
    elif superclass == "ResBlockNet":
        return ResBlockNet(config=model_config)


def create_loss(args):
    loss_name = args["loss"]["name"]
    if loss_name in ["l1", "mae"]:
        loss_func = nn.L1Loss
    elif loss_name in ["l2", "mse"]:
        loss_func = nn.MSELoss
    elif loss_name == "huber":
        loss_func = nn.SmoothL1Loss
    elif loss_name in ["smoothl1", "char"]:
        loss_func = nn.SmoothL1Loss
    elif loss_name == "SSIM":
        # You'll need to define your own custom SSIMLoss class
        # and import it here.
        loss_func = SSIMLoss
    else:
        raise ValueError(f"Invalid loss name: {loss_name}")

    if "args" in args["loss"]:
        args = args["loss"]["args"]
        try:
            loss_func = loss_func(**args)
        except TypeError as e:
            raise ValueError(
                f"Invalid arguments for loss function {loss_name}: {args}"
            ) from e
    else:
        loss_func = loss_func()

    return loss_func


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