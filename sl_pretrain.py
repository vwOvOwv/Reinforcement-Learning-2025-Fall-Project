import os
import random
import argparse
import time
import wandb
import tqdm
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from sl_pretrain.dataset.dataset import MahjongDataset
from sl_pretrain.utils.metric import AverageMeter, evaluate
from rl_ppo.models.network import CNNModel


def set_random_state(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, checkpoint_dir, filename):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    fpath = os.path.join(checkpoint_dir, filename)
    torch.save(state, fpath)

if __name__ == '__main__':
    # get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', 
                        help='The path of config file', required=True)
    parser.add_argument('--seed', type=int, default=2025, 
                        help='Random seed (default: 2025)')
    parser.add_argument('--log', type=int, default=1, choices=[0, 1],
                        help='Log mode: 0 for off, 1 for loss&acc ' \
                        'curve and checkpoints')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)
    if args.config is None:
        raise Exception('Unrecognized config file: {args.config}.')
    else:
        config_path = args.config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # log config info to wandb
    if args.log:
        exp_name = f"{time.strftime('%Y%m%d-%H%M%S')}"
        save_path = os.path.join(config['output_dir'], exp_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        wandb.init(project='RL-2025-Fall-Project',
                   config=config,
                   name=exp_name,
                   dir=save_path)

    # set random state
    set_random_state(args.seed) 

    # set dataset
    dataset = ConcatDataset([MahjongDataset(p) for p in config['data_path']])
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training sample: {train_size}")
    print(f"Validation sample: {val_size}")

    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=config['batch_size'],
                              num_workers=config.get('num_workers', 8),
                              pin_memory=True)
    
    val_loader = DataLoader(val_dataset,
                            shuffle=False,
                            batch_size=config['batch_size'],
                            num_workers=config.get('num_workers', 8),
                            pin_memory=True)
    
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # build model
    model = CNNModel().to(device)

    # set optimizer, scheduler, and loss function
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], 
                          weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, 
                                    T_max=config['epochs'] * len(train_loader))
    criterion = nn.CrossEntropyLoss().to(device)
    
    # save initial checkpoint
    if args.log:
        save_checkpoint(model.state_dict(), checkpoint_dir=save_path, filename='init_ckpt.pt')  # type: ignore
    
    # start training
    iteration = 0
    best_val_acc = 0.0
    for epoch in range(config['epochs']):
        model.train()
        train_loss = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        tqdm_train_loop = tqdm.tqdm(train_loader)

        for idx, (inputs, labels) in enumerate(tqdm_train_loop):
            # inputs, labels = inputs.to(device), labels.to(device)
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            labels = labels.to(device)

            logits, _ = model(inputs)
            optimizer.zero_grad()
            batch_loss = criterion(logits, labels)
            batch_loss.backward()
            optimizer.step()

            train_loss.update(batch_loss.item(), labels.numel())
            acc1, acc5 = evaluate(logits.detach(), labels, topk=(1, 5))
            top1.update(acc1.item(), labels.numel())
            top5.update(acc5.item(), labels.numel())

            tqdm_train_loop.set_postfix(lr=scheduler.get_last_lr()[0], 
                                        loss=train_loss.avg, 
                                        acc1=top1.avg, acc5=top5.avg)
            
            if args.log and iteration % config['log_interval'] == 0:
                wandb.log({
                    "train/loss": train_loss.avg,
                    "train/acc1": top1.avg,
                    "train/acc5": top5.avg,
                    "train/lr": optimizer.param_groups[0]['lr'],
                }, step=iteration)
            
            iteration += 1
            scheduler.step()

        save_checkpoint(model.state_dict(), checkpoint_dir=save_path, filename=f"epoch{epoch + 1}.pt")  # type: ignore

        model.eval()
        with torch.no_grad():
            val_loss = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            for inputs, labels in tqdm.tqdm(val_loader):
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                labels = labels.to(device)
                logits, _ = model(inputs)
                
                batch_loss = criterion(logits, labels)
                val_loss.update(batch_loss.item(), labels.numel())
                acc1, acc5 = evaluate(logits.detach(), labels, topk=(1, 5))
                top1.update(acc1.item(), labels.numel())
                top5.update(acc5.item(), labels.numel())

            if args.log:
                wandb.log({
                    "Epoch": epoch,
                    "val/loss": val_loss.avg,
                    "val/acc1": top1.avg,
                    "val/acc5": top5.avg,
                }, step=iteration)
            print('     Val Loss {loss.avg:.3f}'.format(loss=val_loss))
            print('     Val Acc@1 {top1.avg:.3f}'.format(top1=top1))
            print('     Val Acc@5 {top5.avg:.3f}'.format(top5=top5))
            
            if top1.avg > best_val_acc:
                best_val_acc = top1.avg
                if args.log:
                    save_checkpoint(model.state_dict(), checkpoint_dir=save_path, filename='best.pt')   # type: ignore

    print(f'Train finished')
    print(f'Args: {args}')