import torch 
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP

# train code using multiple gpus
def train(dataloader, model, args):
    if args.distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)
        model = model.cuda(args.local_rank)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif args.data_parallel:
        model = DP(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
    criterion = torch.nn.MSELoss()
    for epoch in range(args.epochs):
        for index, (x, y) in enumerate(dataloader):
            if args.distributed:
                x = x.cuda(args.local_rank, non_blocking=True)
                y = y.cuda(args.local_rank, non_blocking=True)
            else:
                x = x.to(device)
                y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            CosineLR.step()
            if index % 10 == 0:
                print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))
        if epoch % 2 == 0:
            ckpth = "./checkpoints/{}_unet.pth".format(epoch)
            torch.save(model.state_dict(), ckpth)


