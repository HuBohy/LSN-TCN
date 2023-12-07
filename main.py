""" TCN for Laughs and Smiles detection """

import time
from tqdm import tqdm
import json
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import numpy as np

from dataset import NDCME
from utils import get_save_folder

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score

from utilities.metrics import AverageMeter
from models.optim_utils import CosineScheduler
from models.LSNTCN import LSNTCN

def load_args():
    parser = argparse.ArgumentParser(description='Pytorch Lipreading ')
    parser.add_argument('--gpus', default="0", type=str, help="Names of the devices comma separated.")
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('-w', '--num-workers', default=8, type=int, metavar='NW', help='# of workers for dataloading (default: 8)')
    parser.add_argument("--epochs", type=int, default=20, help="number of maximum training epochs")

    parser.add_argument("--dataset", type=str, default="ndc-me", help="the dataset used", choices=['ndc-me'])
    parser.add_argument("--data-video", type=str, default='./datasets/ndc-me/roi', help="video data path")
    parser.add_argument("--data-audio", type=str, default='./datasets/ndc-me/audio', help="audio data path")
    parser.add_argument('--label-path', type=str, default='./datasets/ndc-me/labels/laughs_smiles.txt', help='Path to txt file with labels')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classes')
    parser.add_argument('-m', '--mode', default='fusion', choices=['video', 'audio', 'fusion'], help='choose the modality')

    parser.add_argument('--optim',type=str, default='adam', choices = ['adam','sgd'])
    parser.add_argument('--lr', default=3e-6, type=float, help='initial learning rate')

    parser.add_argument('--config-path', type=str, default='./configs/lrw_resnet18_mstcn.json', help='Model configuration with json format')
    parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")

    parser.add_argument('--logging-dir', type=str, default='./logs', help = 'path to the directory in which to save the log file')

    args = parser.parse_args()
    return args

# torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.benchmark = True

def evaluate(model, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true, y_pred = [], []
    model.eval()
    
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (data_v, data_a, labels) in enumerate(tqdm(test_loader)):
            data_v = data_v.float().to(device)
            data_a = data_a.float().to(device)
            labels = torch.LongTensor(labels).to(device)

            logits = model(data_v, data_a)
            loss = loss_fn(logits, labels)

            pred = logits.argmax(dim=1, keepdim=True)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

        print("{} Confusion Matrix: \n{}".format(args.mode, confusion_matrix(y_true, y_pred, labels=[0, 1, 2])))

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print('Validation balanced accuracy: {:.4f}'.format(bal_acc))
    print('Validation accuracy: {:.4f}'.format(acc))
    print('Validation loss: {:.4f}'.format(loss))
 
    return loss

def train(model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    data_time = AverageMeter()
    batch_time = AverageMeter()
    global_step, epoch = 0, 0

    best_epoch, best_loss = 0, np.inf

    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)

    trainables = [p for p in model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineScheduler(args.lr, args.epochs)

    print('-' * 10)
    epoch += 1
    loss_fn = nn.CrossEntropyLoss()

    print('now training with {:s}, learning rate scheduler: {:s}'.format(args.dataset, scheduler))
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    # saving hyperparameters dict
    with open(f'{args.logging_dir}/hparams.json', 'wt') as f:
        json.dump(vars(args), f, indent=4)

    model.train()
    while epoch < args.epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        for batch_idx, (data_v, data_a, labels) in enumerate(tqdm(train_loader)):
            # measure data loading time
            data_time.update(time.time() - end_time)

            data_v = data_v.float().to(device)
            data_a = data_a.float().to(device)
            labels = torch.LongTensor(labels).to(device)

            optimizer.zero_grad()
            
            logits = model(data_v, data_a)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if batch_idx % args.n_print_steps == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data_v), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss))
            
        print('start validation')
        valid_loss = evaluate(model, test_loader, args)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch

        if best_epoch == epoch:
            torch.save(model.state_dict(), "%s/best_audio_model.pth" % (args.logging_dir))
            torch.save(optimizer.state_dict(), "%s/best_optim_state.pth" % (args.logging_dir))


        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1
        scheduler.adjust_lr(optimizer, epoch)

def main():
    args = load_args()

    video_dirpath = args.data_video
    audio_dirpath = args.data_audio
    
    # -- logging
    args.logging_dir = get_save_folder(args)
    print("Model and log being saved in: {}".format(args.logging_dir))

    model = LSNTCN(num_classes=args.num_classes,
                    relu_type='prelu',
                    mode=args.mode).cuda()

    train_dataset = NDCME(video_dirpath, audio_dirpath, mode='train')
    val_dataset = NDCME(video_dirpath, audio_dirpath, mode='val')
    test_dataset = NDCME(video_dirpath, audio_dirpath, mode='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=WeightedRandomSampler(train_dataset.weighted_sampling, len(train_dataset), replacement=True)
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=WeightedRandomSampler(val_dataset.weighted_sampling, len(val_dataset), replacement=True))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=WeightedRandomSampler(test_dataset.weighted_sampling, len(test_dataset), replacement=True))

    print('Dataset created.')

    train(model, train_loader, val_loader, args)
    evaluate(model, test_loader, args)

if __name__ == '__main__':
    main()
