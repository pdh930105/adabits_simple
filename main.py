import os, sys, shutil, time, random
import argparse


# ----------torch library load ------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import create_loader
import pandas as pd
import numpy as np
import pickle

## import quantized module

from utils import WarmUpLR

import models
from options import Option
from log_utils import make_logger, AverageMeter

import warnings
warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adabits_train(model, train_loader, optimizer, epoch, device, logger, warmup_scheduler=None):
    model.train()
    total_loss = AverageMeter()
    total_acc = AverageMeter()
    total_top5_acc = AverageMeter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if option.adaptive_training == True:
            for bits_idx, bit in enumerate(option.bits_list):
                model.apply(
                    lambda m: setattr(m, 'bits', bit)
                )
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
        else:
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
        
        # train adabits => final bit list result logging
        prec1, prec5 = accuracy(output.cpu().detach(), target.cpu().detach(), (1, 5))
        total_loss.update(loss.item())
        total_acc.update(prec1)
        total_top5_acc.update(prec5)
        optimizer.step()
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        if batch_idx % 100 == 0:
            result_text= 'Train Epoch: [{}] [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()) 
            logger.info(result_text)
    logger.info('Train Epoch: [{}]\t Average Loss: {:.6f}\t Total Acc : {:.4f}\t Total Top5 Acc : {:.4f}'.format(
                epoch, total_loss.avg, total_acc.avg, total_top5_acc.avg))
    print("===="*10)
    
    return total_acc.avg, total_top5_acc.avg, total_loss.avg


def adabits_test(model, test_loader, epoch, device, logger):
    model.eval()
    result_dict = {}
    for bit in option.bits_list:
        result_dict[f'{bit}_total_top1'] = AverageMeter()
        result_dict[f'{bit}_total_top5'] = AverageMeter()
        result_dict[f'{bit}_total_loss'] = AverageMeter()
        
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for bit in option.bits_list:
                model.apply(
                    lambda m : setattr(m, "bits", bit)
                )
                output = model(data)
                test_loss = F.cross_entropy(output, target)
                result_dict[f'{bit}_total_loss'].update(test_loss.item())
                prec1, prec5 = accuracy(output.cpu().detach(), target.cpu().detach(), (1, 5))
                result_dict[f'{bit}_total_top1'].update(prec1)
                result_dict[f'{bit}_total_top5'].update(prec5)
                            
    for bit in option.bits_list:
        logger.info('\nEpoch [{}] bit [{}] Test set: Average loss: {:.4f}, Accuracy: {:.4f}%, Top-5 Accuracy: {:.4f}%\n'.format(
            epoch, bit, result_dict[f'{bit}_total_loss'].avg, result_dict[f'{bit}_total_top1'].avg, result_dict[f'{bit}_total_top5'].avg))
    print("===="*10)
    return result_dict


def main():
    parser = argparse.ArgumentParser(description="resnet test")
    parser.add_argument("--conf_path", type=str, help="hocon config path")
    parser.add_argument("--resume", action="store_true", dest='resume', default=False, help="load pkt and using retraining")
    parser.add_argument("--gpu_num", type=int, default=0, help="select gpu num")

    args = parser.parse_args()
    global option
    option = Option(args.conf_path, args)
    torch.manual_seed(option.seed)
    torch.cuda.manual_seed(option.seed)
    np.random.seed(option.seed)

    for name in models.__dict__:
        if not name.startswith("__") and not callable(models.__dict__[name]):
            models.__dict__[name].set_option(option)
            print("set option : ", name)
            
    
    if option.dataset.lower() == "cifar100":
        cifar100_path = os.path.join(option.data_path, "CIFAR100")
        train_loader, test_loader, n_classes, image_size = create_loader(option.batch_size, cifar100_path, option.dataset)
    elif option.dataset.lower() == "imagenet":
        train_loader ,test_loader, n_classes, image_size = create_loader(option.batch_size, option.data_path, option.dataset)
    
    else : 
        AssertionError("please select dataset cifar100|imagenet")
    

    if option.model_name.lower() in model_names:
        net = models.__dict__[option.model_name.lower()](n_classes)

    else:
        print(option.model_name)
        raise AssertionError("This test only using resnet18im")

    device = torch.device(f'cuda:{args.gpu_num}')
    net = net.to(device)

    
    if option.optimizer.lower() == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=option.lr, momentum=option.momentum, nesterov=option.nesterov)
    
    elif option.optimizer.lower() == "adam":
        optimizer = optim.Adam(net.parameters(), lr= option.lr)
    
    if option.scheduler.lower() == "multi_step":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=option.ml_step, gamma=option.lr_gamma)
    
    if option.warmup > 0:
        warmup_scheduler = WarmUpLR(optimizer, total_iters=option.warmup*len(train_loader))

    if args.resume:
        option.log_override = False
        option.set_save_path()
        checkpoint = torch.load(os.path.join(option.save_path, "last_checkpoint.pth"))
        start_epoch = checkpoint['end_epoch']+1
        
        if start_epoch < 5:
            print("re-train for using warmup train")
            option.log_override = True
            option.set_save_path()
            start_epoch = 0   
        else :
            print(f"load pretrained model : epoch {start_epoch}")
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        print(f"train {option.model_name}")
        start_epoch = 0
        option.log_override = True
        option.set_save_path()

    logger = make_logger("train", os.path.join(option.save_path, "train.log"))
    writer = SummaryWriter(os.path.join(option.save_path, "tfboard_result"))


    random_sampler= torch.utils.data.RandomSampler(test_loader.dataset)
    sample_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=128, sampler=random_sampler)

    check_data, check_label = next(iter(sample_loader))
    check_data = check_data.to(device)
    check_label = check_label.to(device)

    ones_shape = [option.batch_size, 3, 224, 224] if option.dataset.lower() == "imagenet" else [option.batch_size, 3, 32, 32]
    dummy_input = torch.ones(ones_shape).to(device) * 0.1


    with torch.no_grad():
        net.eval()
        writer.add_graph(net, dummy_input)

    #del dummy_input
    #del ones_shape



    best_test_acc = 0
    best_epoch = 0
    save_best_acc_path = os.path.join(option.save_path, "best_checkpoint.pth")

    option.print_parameters()
    for epoch in range(start_epoch, option.epochs):
        logger.info(f"-------{epoch} epoch start-----------")

        if epoch < option.warmup:
            train_acc, train_top5_acc, train_loss = adabits_train(net, train_loader, optimizer, epoch, device, logger, warmup_scheduler)
        else:
            train_acc, train_top5_acc, train_loss = adabits_train(net, train_loader, optimizer, epoch, device, logger)
        print(f"-------{epoch} epoch end  -----------\n")

        scheduler.step()
        logger.info(f"-------{epoch} epoch end-----------")
        
        print("----- test and print accuracy ------------------")
        result_dict=adabits_test(net, test_loader, epoch, device, logger)
        result_loss_dict = {key:value.avg for key, value in result_dict.items() if "loss" in key}
        result_acc_dict = {key:value.avg for key, value in result_dict.items() if "top" in key}
        result_loss_dict.update({"train_loss":train_loss})
        result_acc_dict.update({"train_Acc_Top1" : train_acc})
        result_acc_dict.update({"train_Acc_Top5" : train_top5_acc})
        writer.add_scalars("Loss", result_loss_dict, epoch)
        writer.add_scalars("Accuracy", result_acc_dict, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

        print("----- test end -------------------------")
        print("\n")
        logger.info(f"save intermediate epoch [{epoch}] result\n\n")
        save_state_dict_path = os.path.join(option.save_path, f"last_checkpoint.pth")

        if result_acc_dict[f'{option.bits_list[0]}_total_top1'] > best_test_acc:
            logger.info(f"logging best performance {epoch} epoch")
            print(f"logging best performance {epoch} epoch")
            torch.save(net.state_dict(), save_best_acc_path)
            best_epoch = epoch
            best_test_acc = result_acc_dict[f'{option.bits_list[0]}_total_top1']
            writer.add_scalar("Best Test Acc", best_test_acc, best_epoch)
            
        torch.save({
            'end_epoch': epoch,
            'model_state_dict' : net.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict()
        }, save_state_dict_path)


if __name__ == '__main__':
    main()
