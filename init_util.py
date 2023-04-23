import json
import os
import shutil
from data import ImagePairs
from torch.utils.data import DataLoader
from shrinknet import *
from resblocknet import *
from evnet import *
from util import SSIMLoss

def create_dataloaders(training_dataset, evaluation_dataset, batch_size=16, num_workers=5):
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    evaluation_dataloader = DataLoader(
        dataset=evaluation_dataset, batch_size=1, num_workers=num_workers, pin_memory=True)

    return training_dataloader, evaluation_dataloader

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