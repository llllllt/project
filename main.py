from train import train
import argparse
from data import get_dataloader
import torch
from cnn.model import Unet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--fully_sharded", action="store_true")

    # parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    dataloader = get_dataloader("./dataset 2/input", "./dataset 2/ground_truth",
                                batch_size=8,
                                distributed=args.distributed)
    
    model = Unet(3, 3)

    train(dataloader, model, args)

main()