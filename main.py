from data import ImagePairs
from train import *
from shrinking_based import *
from utils import BestModel

training_dataset = ImagePairs("F:/ImagePairs/training")
evaluation_dataset = ImagePairs("F:/ImagePairs/evaluation")
model = ResNet2(feature_dimension=32, shrinking_filters=5)
best = train(model=model, dataloaders=create_dataloaders(training_dataset, evaluation_dataset, num_workers=0), epochs=10, learning_rate=0.001)
best.save(path="F:/results/ResNet2_test.ser")