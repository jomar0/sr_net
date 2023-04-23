from data import ImagePairs
from train_dist import *
import os
from shrinking_based import *
from torch.nn.parallel import DistributedDataParallel
from utils import BestModel

init_dist_env()
training_dataset = ImagePairs("/home/s.1909943/ImagePairs/training")
evaluation_dataset = ImagePairs("/home/s.1909943/ImagePairs/evaluation")
fsrcnn = FSRCNN(types=["conv", "dws", "dws", "dws"]).cuda()
fsrcnn = DistributedDataParallel(fsrcnn, device_ids=int(os.environ['LOCAL_RANK']))
dataloaders = create_dataloaders(training_dataset=training_dataset, evaluation_dataset=evaluation_dataset)
best = train(fsrcnn, dataloaders, 10)
best.save(path="/home/s.1909943/torch/fsrcnn_test.ser")