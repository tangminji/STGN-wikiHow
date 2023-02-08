# our method main.py
# to discriminate hard&noisy sample
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import csv
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
from tensorboard_logger import log_value
from transformers import get_linear_schedule_with_warmup, AdamW

#from common import utils
from common.utils import log, set_seed, generate_log_dir,generate_save_dir,save_config, AverageMeter, \
    compute_topk_accuracy, checkpoint, log_intermediate_iteration_stats, log_stats, hook_fn_random_walk, quantile
from common.cmd_args_wikihow import args
from dataset.wikihow_dataset import get_WIKIHOW_train_val_test_loader, get_WIKIHOW_model_and_loss_criterion
import json
from hyperopt import STATUS_OK

# TODO multi gpu
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"#"0,1,2,7"#

best_acc = 0

# model and data
MD_CLASSES = {
    'WIKIHOW': (get_WIKIHOW_train_val_test_loader, get_WIKIHOW_model_and_loss_criterion)
}


def validate(args, val_loader, model, criterion, epoch):
    """Evaluates model on validation set and logs score on tensorboard.

    Args:
        args (argparse.Namespace):
        val_loader (torch.utils.data.dataloader): dataloader for validation set.
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss
        epoch (int): current epoch
    """
    global best_acc
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')  # for classification
    MAE = AverageMeter("MAE", ":6.2f")  # for regression
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        # input_ids, input_mask, segment_ids, label_ids, index_dataset
        for i, batch in enumerate(val_loader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            target = batch[3]
            logits, = model(**inputs)
            loss = criterion(logits, target)
            loss = loss.mean()

            # measure accuracy and record loss
            losses.update(loss.item(), inputs["input_ids"].size(0))
            acc1 = compute_topk_accuracy(logits, target, topk=(1,))
            top1.update(acc1[0].item(), inputs["input_ids"].size(0))
        log(args.logpath,'Val-Epoch-{}: Acc:{}, Loss:{}\n'.format(epoch, top1.avg, losses.avg))

    log_value('val/loss', losses.avg, step=epoch)
    # Logging results on tensorboard

    log_value('val/accuracy', top1.avg, step=epoch)
    # Save checkpoint.


    acc = top1.avg
    if acc > best_acc:
        best_acc = acc
        print('Saving..')
        model_to_save = model.module if hasattr(model, "module") else model

        model_output_file = args.save_dir + '/epoch_{}/model'.format(epoch)
        os.makedirs(model_output_file, exist_ok=True)
        model_to_save.save_pretrained(model_output_file)
        # os.system("cp %s %s" % (os.path.join(args.roberta_path, "merges.txt"), model_output_file))
        # os.system("cp %s %s" % (os.path.join(args.roberta_path, "vocab.json"), model_output_file))
        for last_e in range(epoch):
            last_e_path = args.save_dir + '/epoch_{}/model'.format(last_e)
            if os.path.exists(last_e_path):
                shutil.rmtree(last_e_path, ignore_errors=True)

    state = {
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    os.makedirs(args.save_dir + '/epoch_{}'.format(epoch),exist_ok=True)
    torch.save(obj=state, f=args.save_dir + '/epoch_{}/state'.format(epoch))
    return losses.avg, top1.avg


def test(args, test_loader, model, criterion, epoch):
    """Evaluates model on test set and logs score on tensorboard.

    Args:
        args (argparse.Namespace):
        test_loader (torch.utils.data.dataloader): dataloader for test set.
        model (torch.nn.Module):
        criterion (torch.nn.modules.loss): cross entropy loss
        epoch (int): current epoch
    """
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')  # for classification
    MAE = AverageMeter("MAE", ":6.2f")  # for regression
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        # input_ids, input_mask, segment_ids, label_ids, index_dataset
        for i, batch in enumerate(test_loader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            target = batch[3]
            logits, = model(**inputs)
            loss = criterion(logits, target)
            loss = loss.mean()

            # measure accuracy and record loss
            losses.update(loss.item(), inputs["input_ids"].size(0))
            acc1 = compute_topk_accuracy(logits, target, topk=(1,))
            top1.update(acc1[0].item(), inputs["input_ids"].size(0))
        log(args.logpath,'Test-Epoch-{}: Acc:{}, Loss:{}\n'.format(epoch, top1.avg, losses.avg))

    log_value('test/loss', losses.avg, step=epoch)
    # Logging results on tensorboard

    log_value('test/accuracy', top1.avg, step=epoch)
    return losses.avg, top1.avg


def train_for_one_epoch(args,
                        train_loader_iter,
                        model,
                        criterion,
                        optimizer,
                        epoch,
                        global_iter,
                        scheduler=None,
                        ):
    """Train model for one epoch on the train set.

    Args:
        args (argparse.Namespace):
        train_loader (torch.utils.data.dataloader): dataloader for train set.
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss.
        optimizer : optimizer for model parameters.
        epoch (int): current epoch.
        global_iter (int): current iteration count.

    Returns:
        global iter (int): updated iteration count after 1 epoch.
    """
    # loss-for-each-sample
    # Initialize counters
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')  # for classification
    
    # Switch to train mode
    model.train()
    start_epoch_time = time.time()
    # all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_index_dataset
    loss_lst = []
    is_train_loader_iter_empty = False
    print("start train one epoch ... ")
    for i in range(args.sub_epochs_step):
        try:
            batch = next(train_loader_iter)
        except StopIteration:
            is_train_loader_iter_empty = True
            break
        global_iter = global_iter + 1

        # print("cudaing batch ... ")
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }

        target, index = batch[3:5] #record index
        args.index = index
        # print("index_dataset_in_train_loop",index_dataset)
        # Flush the gradient buffer for model and data-parameters
        optimizer.zero_grad()
        # Compute logits
        # print("cal model ... ")
        output, = model(**inputs)
        args.sm = F.softmax(output)

 
        # SLN
        if args.mode == 'GN_on_label':
            onehot = F.one_hot(target.long(), args.num_class).float()
            onehot += args.sigma*torch.randn(onehot.size()).to(args.device) # add Gauss Noise
            loss = -torch.sum(F.log_softmax(output, dim=1)*onehot, dim=1) # Cross Entropy
        else:

            if args.mode == 'Random_walk':
                output.register_hook(hook_fn_random_walk)
            loss = criterion(output, target)
            if args.mode == 'Random_walk':
                # TODO: element1: from loss perspective
                # TODO: quantile
                loss_lst.append(loss.detach().cpu().numpy().tolist())
                if len(loss_lst) > args.avg_steps:
                    loss_lst.pop(0)
                #print('random_walk',len(loss_lst[-1]),args.drop_rate_schedule[args.cur_epoch - 1])
                losses_lst = sum(loss_lst,[])

                k1 = quantile(torch.tensor(losses_lst).to(args.device),
                                    1 - args.drop_rate_schedule[args.cur_epoch - 1])
                #TODO: element2: from forgetting events perspective, see algorithm 1 in ICLR19 an empirical study of example...
                _, predicted = torch.max(output.data, 1)
                # Update statistics and loss
                acc = (predicted == target).to(torch.long)
                forget_or_not = torch.gt(args.prev_acc[index], acc)#greater than
                args.forgetting[index] = args.forgetting[index] + forget_or_not
                args.prev_acc[index] = acc

                #when to update, since forgetting times of any sample reaches to args.forget_times

                # if (args.forgetting>args.forget_times).any():
                times_ge_or_not = torch.ge(args.forgetting[index], args.forget_times).detach()
                if times_ge_or_not.any(): 
                # if args.forget_times in args.forgetting: 
                #     #greater or equal
                #     times_ge_or_not = torch.ge(args.forgetting[index], args.forget_times).detach()
                    args.sign_forgetting_events = ((1-args.ratio_l)*args.total) * torch.tensor([1 if t == True else -1 for t in times_ge_or_not]).to(args.device)
                    args.sign_loss = (args.ratio_l * args.total) * torch.sign(loss - k1).to(args.device)
                else:
                    args.sign_forgetting_events = torch.tensor([0]*len(loss)).to(args.device)
                    if args.ratio_l != 0:
                        args.sign_loss = torch.sign(loss - k1).to(args.device)
                    else:
                        args.sign_loss = torch.tensor([0] * len(loss)).to(args.device)

        loss = loss.mean()

        # Compute gradient and do SGD step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2)
        optimizer.step()
        scheduler.step()
        
        args.sm = None

        # Measure accuracy and record loss
        losses.update(loss.item(), inputs["input_ids"].size(0))

        acc1 = compute_topk_accuracy(output, target, topk=(1,))
        top1.update(acc1[0].item(), inputs["input_ids"].size(0))

        # Log stats for data parameters and loss every few iterations
        if i % args.print_freq == 0:
            log_intermediate_iteration_stats(epoch,global_iter,losses,top1=top1)

    # Print and log stats for the epoch
    log(args.logpath,'Time for epoch: {}\n'.format(time.time() - start_epoch_time))
    log_value('train/loss', losses.avg, step=epoch)

    log(args.logpath,'Train-Epoch-{}: Acc:{}, Loss:{}\n'.format(epoch, top1.avg, losses.avg))
    log_value('train/accuracy', top1.avg, step=epoch)

    train_metric = top1.avg
    return global_iter, losses.avg, train_metric, is_train_loader_iter_empty  # top1.avg


def main_worker(args):
    """Trains model on ImageNet using data parameters
    Args:
        args (argparse.Namespace):
        config1 (dict): config file for the experiment.
    """
    global best_acc
    global_iter = 0
    # learning_rate_schedule = np.array([80, 100, 160])
    loaders, mdl_loss = MD_CLASSES[args.dataset]
    # Create model
    model, loss_criterion, loss_criterion_val, logname = mdl_loss(args)

    # Define optimizer

    param_mo = model.module if hasattr(model, "module") else model
    optimizer = AdamW(param_mo.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_epsilon)

    # Get train and validation dataset loader
    train_loader, val_loader, test_loader = loaders(args)

    train_length = len(train_loader.dataset)
    args.sigma_dyn = torch.tensor([args.sigma]*train_length,
                           dtype=torch.float32,
                           requires_grad=False,
                           device=args.device)

    args.prev_acc = torch.tensor(np.zeros(train_length),
                           dtype=torch.long,
                           requires_grad=False,
                           device=args.device)
    args.forgetting = torch.tensor(np.zeros(train_length),
                                 dtype=torch.long,
                                 requires_grad=False,
                                 device=args.device)
    args.drop_rate_schedule = np.ones(args.epochs) * args.noise_rate
    args.drop_rate_schedule[:args.num_gradual] = np.linspace(0, args.noise_rate, args.num_gradual)

    # Training loop
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            assert args.task_type == 'classification'
            logwriter.writerow(
                ['epoch', 'train loss', 'train acc', 'val loss', 'val acc', 'test loss', 'test acc'])
    start_epoch_time = time.time()
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_loader) * args.num_train_epochs
    )
    epoch = -1
    best_test = -1
    for num_train_e in range(args.start_epoch, args.num_train_epochs):
        args.cur_epoch = num_train_e+1
        train_loader_iter = iter(train_loader)
        is_train_loader_iter_empty = False
        while not is_train_loader_iter_empty:
            epoch += 1
            # Train for one epoch
            global_iter, train_loss, train_metric, is_train_loader_iter_empty = train_for_one_epoch(
                args=args,
                train_loader_iter=train_loader_iter,
                model=model,
                criterion=loss_criterion,
                optimizer=optimizer,
                epoch=epoch,
                global_iter=global_iter,
                scheduler=scheduler)

            # Evaluate on validation set
            val_loss, val_metric = validate(args, val_loader, model, loss_criterion_val, epoch)
            # Evaluate on test set
            test_loss, test_metric = test(args, test_loader, model, loss_criterion_val, epoch)

            if val_metric == best_acc:
                best_test = test_metric

            # log model metrics
            with open(logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([epoch, train_loss, train_metric, val_loss, val_metric, test_loss, test_metric])
    run_time = time.time() - start_epoch_time
    # record best_acc
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([-1, -1, -1, -1, best_acc, -1, best_test])
    
    # Val_best Test_at_val_best
    with open(os.path.join(args.log_dir, 'best_results.txt'), 'w') as outfile:
        outfile.write(f'{best_acc}\t{best_test}') #Val Test
    log(args.logpath, '\nBest ACC: {}\t{}\n'.format(best_acc,best_test))
    log(args.logpath, 'Total Time: {}\n'.format(run_time))
    loss = - best_test
    return {'loss': loss, 'best_acc': best_acc, 'test_at_best': best_test,
            'params': params, 'train_time': run_time, 'status': STATUS_OK}


def main(params):
    if 'STGN' in args.exp_name:
        #TODO: automatic adjustment (sig_max, lr_sig)
        args.times = params['times']
        args.sigma = params['sigma']
        args.sig_max = 2.0 * params['sigma']
        args.lr_sig = 0.1 * params['sigma']
        #others
        args.avg_steps = params['avg_steps']
        args.ratio_l = params['ratio_l']
        args.noise_rate = params['noise_rate']
        args.forget_times = params['forget_times']

    if 'SLN' in args.exp_name:
        args.sigma = params['sigma']
    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)
    args.logpath = os.path.join(args.exp_name, 'log.txt')
    args.log_dir = os.path.join(args.exp_name)
    args.save_dir = os.path.join(args.exp_name,'weights')


    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    generate_log_dir(args)
    generate_save_dir(args)

    log(args.logpath, 'Settings: {}\n'.format(args))


    # Set seed for reproducibility
    set_seed(args)
    # Simply call main_worker function
    res = main_worker(args)
    
    return res


if __name__ == '__main__':
    print("load params from : ", args.params_path)
    params = json.load(open(args.params_path, 'r', encoding="utf-8"))['best'] if 'STGN' in args.exp_name else {}
    if 'SLN' in args.exp_name:
        params = {'sigma': args.sigma}
    
    # STGN sigma 0.0005, 0.001
    # times 30 
    # noise_rate 0.2~0.4


    assert params is not None
    res = main(params=params)
