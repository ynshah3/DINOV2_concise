import torch
from torch.utils.data import DataLoader
from imagenet import imagenet_dataset
from torch.nn.utils import prune
import logging
from experiment import Experiment


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename='healthy_eval_log.txt', filemode="a+")
    
    train_data, val_data = imagenet_dataset()
    train_loader = DataLoader(train_data, batch_size=128, num_workers=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, num_workers=8, shuffle=False)
    
    exp = Experiment()
    exp.test(val_loader)
