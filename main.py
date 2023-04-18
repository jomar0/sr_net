from data import ImagePairs
from train import *
from shrinking_based import *
import os
import time
from utils import BestModel, create_dataloaders
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Trainer for SRNET")

# Add the arguments
parser.add_argument("name", type=str, help="Name of the program")
parser.add_argument("model_name", type=str, help="Name of the model")
parser.add_argument("feature_dimension", type=int, help="Dimension of the feature space")
parser.add_argument("shrinking_filters", type=int, help="Number of shrinking filters")
parser.add_argument("mapping_depth", type=int, help="mapping depth")
parser.add_argument("epochs", type=int, help="Number of training epochs")
parser.add_argument("learning_rate", type=float, help="Learning rate for the optimizer")
parser.add_argument("batch_size", type=int, help="Batch size")
parser.add_argument("types", nargs="+", type=str, help="List of types")

# Parse the arguments
args = parser.parse_args()

# Access the arguments
name = args.name
model_name = args.model_name
feature_dimension = args.feature_dimension
shrinking_filters = args.shrinking_filters
mapping_depth= args.mapping_depth
epochs = args.epochs
learning_rate = args.learning_rate
batch_size= args.batch_size
types = args.types

training_dataset = ImagePairs("/home/u1909943/ImagePairs/training")
evaluation_dataset = ImagePairs("/home/u1909943/ImagePairs/evaluation")

if model_name == "FSRCNN":
    model = FSRCNN(feature_dimension=feature_dimension, shrinking_filters=shrinking_filters, mapping_depth=mapping_depth, types=types)
elif model_name == "ResNet1":
    model = ResNet1(feature_dimension=feature_dimension, shrinking_filters=shrinking_filters,mapping_depth=mapping_depth, types=types)
elif model_name == "ResNet2":
    model = ResNet2(feature_dimension=feature_dimension, shrinking_filters=shrinking_filters,mapping_depth=mapping_depth, types=types)
else:
    raise Exception("Not a valid model")

log_path=f"/home/u1909943/MSc/results/{name}.log"
if not os.path.exists(f"/home/u1909943/MSc/results/{name}/"):
    os.makedirs(f"/home/u1909943/MSc/results/{name}/")

# file.write the arguments
with open(log_path, "a") as file:
    file.write("="*80 + "\n")
    file.write(f"Log File for Training Session {name}\n")
    file.write(f"Name: {name}\n")
    file.write(f"Model name: {model_name}\n")
    file.write(f"Feature dimension: {feature_dimension}\n")
    file.write(f"Shrinking filters: {shrinking_filters}\n")
    file.write(f"Mapping Depth: {mapping_depth}\n")
    file.write(f"Epochs: {epochs}\n")
    file.write(f"Learning rate: {learning_rate}\n")
    file.write(f"Batch Size: {batch_size}\n")
    file.write(f"Types: {types}\n")
    file.write("="*80 + "\n")

print("="*80)
print(f"Training Session {name}")
print(f"Name: {name}")
print(f"Model name: {model_name}")
print(f"Feature dimension: {feature_dimension}")
print(f"Mapping Depth: {mapping_depth}")
print(f"Shrinking filters: {shrinking_filters}")
print(f"Epochs: {epochs}")
print(f"Learning rate: {learning_rate}")
print(f"Batch Size: {batch_size}")
print(f"Types: {types}")
print("="*80)


start = time.time()
best = train(model=model, dataloaders=create_dataloaders(training_dataset, evaluation_dataset, num_workers=0, batch_size=batch_size), epochs=epochs, learning_rate=learning_rate, 
log_path=log_path)

elapsed = time.time() - start
formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed))

model.cpu()
model.load_state_dict(best.best_model_psnr)

psnr_params = sum(
	param.numel() for param in model.parameters()
)

model.load_state_dict(best.best_model_ssim)
ssim_params = sum(
	param.numel() for param in model.parameters()
)


with open(log_path, "a") as file:
    file.write(f"\n\nTook {formatted_time}\n")
    file.write(f"Best Epoch for PSNR: {best.best_epoch_psnr} - PSNR: {best.best_psnr} -  # of Parameters: {psnr_params}\n")
    file.write(f"Best Epoch for SSIM: {best.best_epoch_ssim} - SSIM: {best.best_ssim} -  # of Parameters: {ssim_params}\n")

print(f"Took {formatted_time}")
print(f"Best Epoch for PSNR: {best.best_epoch_psnr} - PSNR: {best.best_psnr} -  # of Parameters: {psnr_params}")
print(f"Best Epoch for SSIM: {best.best_epoch_ssim} - SSIM: {best.best_ssim} -  # of Parameters: {ssim_params}")

best.save(path=f"/home/u1909943/MSc/results/{name}/{name}.model")

