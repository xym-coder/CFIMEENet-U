import os
import gc
import time
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler
from torch.utils.data import ConcatDataset
from dataloader import TotalText, Ctw1500Text, Uyghur
from model.loss import TextLoss
from model.textnet import TextNet
from tool_functions.augmentation import Augmentation, BaseTransform
from config.config import config as cfg, update_config, logging_config
from tool_functions.base_tool_functions import AverageMeter
from tool_functions.base_tool_functions import mkdirs, to_device, rescale_result, get_cosine_map
from config.commandline_parameters import BaseOptions
from tool_functions.visualize import visualize_network_output
from tool_functions.scheduler import FixLR
from tool_functions.eval import deal_eval_total_text, deal_eval_ctw1500, deal_eval_uyghur, data_transfer_uyghur
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import glob                 
import logging
from datetime import datetime
import subprocess
import platform
import argparse
import cv2
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

lr = None, train_step = 0, best_loss = float('inf'), best_epoch = 0, best_f1 = 0.0, best_f1_epoch = 0, best_f1_precision = 0.0, best_f1_recall = 0.0, best_distance_loss = float('inf'), best_distance_epoch = 0, best_point_loss = float('inf'), best_point_epoch = 0, best_energy_loss = float('inf')
best_energy_epoch = 0, best_cls_loss = float('inf'), best_cls_epoch = 0, best_dir_loss = float('inf'), best_dir_epoch = 0, writer = None

def setup_logging(exp_name):
    global writer
    log_dir = '/root/CFIMEEMNet-U/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{exp_name}_{timestamp}.txt')

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    tensorboard_dir = f'/root/CFIMEEMNet-U/runs/{exp_name}_{timestamp}'
    writer = SummaryWriter(tensorboard_dir)
    logging.info(f'TensorBoard logs will be saved to: {tensorboard_dir}')
    logging.info('To view TensorBoard, run: tensorboard --logdir=/root/CFIMEEMNet-U/runs')

def save_model(model, epoch, lr, optimzer, scheduler, is_best=False, is_best_f1=False,
               is_best_distance=False, is_best_point=False, is_best_energy=False,
               is_best_midline=False, is_best_cls=False, is_best_dir=False):

    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict() if not cfg.mgpu else model.module.state_dict(),
        'optimizer': optimzer.state_dict(),             
        'scheduler': scheduler.state_dict(),       
        'best_loss': best_loss if is_best else None,
        'best_epoch': epoch if is_best else None,
        'best_f1': best_f1 if is_best_f1 else None,
        'best_f1_epoch': best_f1_epoch if is_best_f1 else None,
        'best_distance_loss': best_distance_loss if is_best_distance else None,
        'best_distance_epoch': best_distance_epoch if is_best_distance else None,
        'best_point_loss': best_point_loss if is_best_point else None,
        'best_point_epoch': best_point_epoch if is_best_point else None,
        'best_energy_loss': best_energy_loss if is_best_energy else None,
        'best_energy_epoch': best_energy_epoch if is_best_energy else None,
        'best_midline_loss': best_midline_loss if is_best_midline else None,
        'best_midline_epoch': best_midline_epoch if is_best_midline else None,
        'best_cls_loss': best_cls_loss if is_best_cls else None,
        'best_cls_epoch': best_cls_epoch if is_best_cls else None, 
        'best_dir_loss': best_dir_loss if is_best_dir else None,
        'best_dir_epoch': best_dir_epoch if is_best_dir else None, 
    }
    if is_best: 
        save_path = os.path.join(save_dir, 'CFIMEEMNet-U_{}_best_total.pth'.format(model.backbone_name))
        torch.save(state_dict, save_path)
        logging.info('Saving best total loss model to {}.'.format(save_path))  

    if is_best_f1:
        save_path = os.path.join(save_dir, f'CFIMEEMNet-U_{model.backbone_name}_best_f1.pth') 
        torch.save(state_dict, save_path)
        logging.info('Saving best F1 model to {}.'.format(save_path))  

    if is_best_distance:
        save_path = os.path.join(save_dir, f'CFIMEEMNet-U_{model.backbone_name}_best_distance.pth') 
        torch.save(state_dict, save_path)
        logging.info('Saving best distance loss model to {}.'.format(save_path))

    if is_best_point:
        save_path = os.path.join(save_dir, f'CFIMEEMNet-U_{model.backbone_name}_best_point.pth')
        torch.save(state_dict, save_path)
        logging.info('Saving best point loss model to {}.'.format(save_path))

    if is_best_energy:
        save_path = os.path.join(save_dir, f'CFIMEEMNet-U_{model.backbone_name}_best_energy.pth')
        torch.save(state_dict, save_path)
        logging.info('Saving best energy loss model to {}.'.format(save_path))

    if is_best_midline:
        save_path = os.path.join(save_dir, f'CFIMEEMNet-U_{model.backbone_name}_best_midline.pth')  
        torch.save(state_dict, save_path)
        logging.info('Saving best midline loss model to {}.'.format(save_path))

    if is_best_cls:
        save_path = os.path.join(save_dir, f'CFIMEEMNet-U_{model.backbone_name}_best_cls.pth')  
        torch.save(state_dict, save_path)
        logging.info('Saving best classification loss model to {}.'.format(save_path))

    if is_best_dir:
        save_path = os.path.join(save_dir, f'CFIMEEMNet-U_{model.backbone_name}_best_dir.pth')  
        torch.save(state_dict, save_path)
        logging.info('Saving best direction loss model to {}.'.format(save_path))

    save_path = os.path.join(save_dir, 'CFIMEEMNet-U_{}_{}.pth'.format(model.backbone_name, epoch))  
    torch.save(state_dict, save_path)

    current_checkpoint = os.path.basename(save_path)
    best_checkpoints = [
        'CFIMEEMNet-U_{}_best_total.pth'.format(model.backbone_name),
        f'CFIMEEMNet-U_{model.backbone_name}_best_f1.pth',
        f'CFIMEEMNet-U_{model.backbone_name}_best_distance.pth',
        f'CFIMEEMNet-U_{model.backbone_name}_best_point.pth',
        f'CFIMEEMNet-U_{model.backbone_name}_best_energy.pth',
        f'CFIMEEMNet-U_{model.backbone_name}_best_cls.pth',
        f'CFIMEEMNet-U_{model.backbone_name}_best_dir.pth'
    ]  
    checkpoint_pattern = os.path.join(save_dir, 'CFIMEEMNet-U_{}_*.pth'.format(model.backbone_name))  

    for checkpoint in glob.glob(checkpoint_pattern):
        checkpoint_name = os.path.basename(checkpoint)
        if checkpoint_name not in [current_checkpoint] + best_checkpoints:
            try:
                os.remove(checkpoint)
                logging.info(f'Removed old checkpoint: {checkpoint}')
            except OSError as e:
                logging.info(f'Error removing checkpoint {checkpoint}: {e}')

def load_model(model, model_path, optimizer=None, scheduler=None):
    logging.info('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path)
    try:
        model.load_state_dict(state_dict['model'])
    except RuntimeError as e:
        logging.info("Missing key in state_dict, try to load with strict = False")
        model.load_state_dict(state_dict['model'], strict = False)
        logging.info(e)

    if optimizer is not None and 'optimizer' in state_dict:
        optimizer.load_state_dict(state_dict['optimizer'])
        logging.info("Optimizer state loaded")
    if scheduler is not None and 'scheduler' in state_dict:
        scheduler.load_state_dict(state_dict['scheduler'])
        logging.info("Scheduler state loaded")
    if 'best_loss' in state_dict and state_dict['best_loss'] is not None:
        global best_loss    
        best_loss = state_dict['best_loss']
        logging.info(f"Loaded best_loss: {best_loss:.4f}")
    if 'best_epoch' in state_dict and state_dict['best_epoch'] is not None:
        global best_epoch
        best_epoch = state_dict['best_epoch']
        logging.info(f"Loaded best_epoch: {best_epoch}")
    if 'best_f1' in state_dict and state_dict['best_f1'] is not None: 
        global best_f1                    
        best_f1 = state_dict['best_f1']
        logging.info(f"Loaded best_f1: {best_f1:.4f}")
    if 'best_f1_epoch' in state_dict and state_dict['best_f1_epoch'] is not None:
        global best_f1_epoch               
        best_f1_epoch = state_dict['best_f1_epoch']
        logging.info(f"Loaded best_f1_epoch: {best_f1_epoch}")

    if 'best_distance_loss' in state_dict and state_dict['best_distance_loss'] is not None:
        best_distance_loss = state_dict['best_distance_loss']
        logging.info(f"Loaded best_distance_loss: {best_distance_loss:.4f}")

    if 'best_distance_epoch' in state_dict and state_dict['best_distance_epoch'] is not None: 
        best_distance_epoch = state_dict['best_distance_epoch']
        logging.info(f"Loaded best_distance_epoch: {best_distance_epoch}")

    if 'best_point_loss' in state_dict and state_dict['best_point_loss'] is not None: 
        best_point_loss = state_dict['best_point_loss']
        logging.info(f"Loaded best_point_loss: {best_point_loss:.4f}")

    if 'best_point_epoch' in state_dict and state_dict['best_point_epoch'] is not None:
        best_point_epoch = state_dict['best_point_epoch']
        logging.info(f"Loaded best_point_epoch: {best_point_epoch}")

    if 'best_energy_loss' in state_dict and state_dict['best_energy_loss'] is not None: 
        best_energy_loss = state_dict['best_energy_loss']
        logging.info(f"Loaded best_energy_loss: {best_energy_loss:.4f}")

    if 'best_energy_epoch' in state_dict and state_dict['best_energy_epoch'] is not None:
        best_energy_epoch = state_dict['best_energy_epoch']
        logging.info(f"Loaded best_energy_epoch: {best_energy_epoch}")

    if 'best_midline_loss' in state_dict and state_dict['best_midline_loss'] is not None: 
        best_midline_loss = state_dict['best_midline_loss']
        logging.info(f"Loaded best_midline_loss: {best_midline_loss:.4f}")

    if 'best_midline_epoch' in state_dict and state_dict['best_midline_epoch'] is not None: 
        best_midline_epoch = state_dict['best_midline_epoch']
        logging.info(f"Loaded best_midline_epoch: {best_midline_epoch}")

    if 'best_cls_loss' in state_dict and state_dict['best_cls_loss'] is not None: 
        best_cls_loss = state_dict['best_cls_loss']
        logging.info(f"Loaded best_cls_loss: {best_cls_loss:.4f}")

    if 'best_cls_epoch' in state_dict and state_dict['best_cls_epoch'] is not None:
        best_cls_epoch = state_dict['best_cls_epoch']
        logging.info(f"Loaded best_cls_epoch: {best_cls_epoch}")

    if 'best_dir_loss' in state_dict and state_dict['best_dir_loss'] is not None:
        best_dir_loss = state_dict['best_dir_loss']
        logging.info(f"Loaded best_dir_loss: {best_dir_loss:.4f}")

    if 'best_dir_epoch' in state_dict and state_dict['best_dir_epoch'] is not None: 
        best_dir_epoch = state_dict['best_dir_epoch']
        logging.info(f"Loaded best_dir_epoch: {best_dir_epoch}")

    if 'best_f1_precision' in state_dict and state_dict['best_f1_precision'] is not None: 
        best_f1_epoch = state_dict['best_f1_precision']
        logging.info(f"Loaded best_f1_epoch: {best_f1_precision}")
    if 'best_f1_recall' in state_dict and state_dict['best_f1_recall'] is not None:
        best_f1_epoch = state_dict['best_f1_recall']
        logging.info(f"Loaded best_f1_epoch: {best_f1_recall}")

    return state_dict.get('epoch', 0)

def _parse_data(inputs):
    input_dict = {}
    inputs = list(map(lambda x: to_device(x), inputs))
    input_dict['img'] = inputs[0]
    input_dict['train_mask'] = inputs[1]
    input_dict['tr_mask'] = inputs[2]
    input_dict['distance_field'] = inputs[3]
    input_dict['direction_field'] = inputs[4]
    input_dict['weight_matrix'] = inputs[5]
    input_dict['gt_points'] = inputs[6]
    input_dict['proposal_points'] = inputs[7]
    input_dict['ignore_tags'] = inputs[8]
    if cfg.embed:
        input_dict['edge_field'] = inputs[9]
    if cfg.mid:
        input_dict['gt_mid_points'] = inputs[9]
        input_dict['edge_field'] = inputs[10]

    input_dict['midline_points'] = inputs[11]   

    return input_dict

def parse_eval_results(fid_path):
    try:
        with open(fid_path, 'r') as f:
            content = f.read()
        import re
        p_match = re.search(r'p:\s*([\d.]+)', content)
        r_match = re.search(r'r:\s*([\d.]+)', content)
        f_match = re.search(r'f:\s*([\d.]+)', content)
        
        if p_match and r_match and f_match:
            precision = float(p_match.group(1))
            recall = float(r_match.group(1))
            f1 = float(f_match.group(1))
            return precision, recall, f1
            
        logging.warning(f"No valid results found in {fid_path}, returning default values")
        return 0.0, 0.0, 0.0
    except Exception as e:
        logging.error(f"Error parsing {fid_path}: {str(e)}")
        return 0.0, 0.0, 0.0

def validate(model, val_loader, output_dir, epoch, dataset_type, optimizer, scheduler):
    global best_f1, best_f1_epoch, best_f1_precision, best_f1_recall, writer
    
    logging.info(f'Starting validation for epoch {epoch} on {dataset_type}')

    if not os.path.exists(output_dir):
        mkdirs(output_dir)

    model.eval()
    total_time = 0.0
    total_frame = 0.0
    contours_list = []
    meta_list = []
    
    with torch.no_grad():
        for i, (image, meta) in enumerate(tqdm(val_loader, desc=f'Validation Epoch {epoch}', total=len(val_loader))):
            input_dict = {'img': to_device(image)}
            idx = 0
            H, W = meta['Height'][idx].item(), meta['Width'][idx].item()
            start = time.time()
            output_dict = model(input_dict)
            torch.cuda.synchronize()
            end = time.time()
            total_time += end - start
            total_frame += image.size(0)
            img_show = image[idx].permute(1, 2, 0).cpu().numpy()
            img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)
            contours = output_dict["py_preds"][-1].int().cpu().numpy()
            img_show, contours = rescale_result(img_show, contours, H, W)
            contours_list.append(contours)
            meta_list.append(meta)

    for contours, meta in zip(contours_list, meta_list):
        idx = 0
        fname = meta['image_id'][idx].replace('JPG', 'txt').replace('jpg', 'txt')
        res_file = os.path.join(output_dir, fname)
        if dataset_type in ['Totaltext', 'Totaltext_mid']:
            if len(contours) == 0:
                logging.info(f"Skipping empty contours for {meta['image_id'][idx]}")
                with open(res_file, 'w') as f:
                    pass
                continue
            with open(res_file, 'w') as f:
                for cont in contours:
                    cont = np.stack([cont[:, 0], cont[:, 1]], 1)
                    if cv2.contourArea(cont) <= 0:
                        continue
                    cont = cont.flatten().astype(str).tolist()
                    cont = ','.join(cont)
                    f.write(cont + '\n')
        elif dataset_type in ['Ctw1500', 'Ctw1500_mid']:
            with open(res_file, 'w') as f:
                for cont in contours:
                    cont = np.stack([cont[:, 0], cont[:, 1]], 1)
                    if cv2.contourArea(cont) <= 0:
                        continue
                    cont = cont.flatten().astype(str).tolist()
                    cont = ','.join(cont)
                    f.write(cont + '\n')
        elif dataset_type in ['Uyghur']:
            contours = data_transfer_uyghur(contours)
            with open(res_file, 'w') as f:
                for cont in contours:
                    cont = cont.flatten().astype(str).tolist()
                    cont = ','.join(cont)
                    f.write(cont + '\n')
        else:
            logging.error(f"Unsupported dataloader type: {dataset_type}")
            return 0.0, 0.0, 0.0, 0.0

    try:
        eval_dir = os.path.join(cfg.output_dir, "Analysis", "output_eval")
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        if dataset_type in ['Totaltext', 'Totaltext_mid']:
            fid_path = f'{eval_dir}/Eval_TotalText_0.7_0.6.txt'
            deal_eval_total_text(debug=False)
            precision, recall, f1 = parse_eval_results(fid_path)
        elif dataset_type in ['Ctw1500', 'Ctw1500_mid']:
            fid_path = f'{eval_dir}/Eval_ctw1500_0.5.txt'
            deal_eval_ctw1500(debug=False)
            precision, recall, f1 = parse_eval_results(fid_path)
        elif dataset_type in ['Uyghur']:
            fid_path = f'{eval_dir}/Eval_Uyghur.txt'
            input_dir = os.path.abspath(os.path.join(cfg.output_dir, cfg.exp_name))
            subprocess.call([
                    'python', 'dataloader/uyghur/Evaluation_Protocol/script.py',
                    '-g', 'dataloader/uyghur/labels_Uyghur.zip',
                    '-s', f'{input_dir}/submit.zip'
                ])
            precision, recall, f1 = parse_eval_results(fid_path)
        else:
            logging.error(f"No evaluation function for dataloader: {dataset_type}")
            return 0.0, 0.0, 0.0, 0.0

        logging.info(f"Validation Epoch {epoch} ({dataset_type}): Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_f1_epoch = epoch
            best_f1_precision = precision
            best_f1_recall = recall
            is_best_f1 = True
            save_model(model, epoch, lr, optimizer, scheduler, is_best_f1=True)
            logging.info(f'New best F1 model saved with F1: {best_f1:.4f}, Precision: {best_f1_precision:.4f}, Recall: {best_f1_recall:.4f} at epoch {best_f1_epoch}')

        if writer is not None:
            writer.add_scalar('Validation/Precision', precision, epoch)
            writer.add_scalar('Validation/Recall', recall, epoch)
            writer.add_scalar('Validation/F1', f1, epoch)
            writer.add_scalar('Validation/FPS', fps, epoch)

            writer.add_scalar('Best/F1', best_f1, epoch)
            writer.add_scalar('Best/Precision', best_f1_precision, epoch)
            writer.add_scalar('Best/Recall', best_f1_recall, epoch)

        return precision, recall, f1, fps
    except Exception as e:
        logging.error(f'Validation failed: {str(e)}')
        return 0.0, 0.0, 0.0, 0.0

def train(model, train_loader, criterion, scheduler, optimizer, epoch, val_loader, dataset_type):

    global train_step, best_loss, best_epoch, writer 
    global best_distance_loss, best_distance_epoch, best_point_loss, best_point_epoch  
    global best_energy_loss, best_energy_epoch, best_midline_loss, best_midline_epoch  
    global best_cls_loss, best_cls_epoch, best_dir_loss, best_dir_epoch 

    losses = AverageMeter()
    distance_losses = AverageMeter()   
    point_losses = AverageMeter()  
    energy_losses = AverageMeter()  
    midline_losses = AverageMeter()  
    cls_losses = AverageMeter()  
    dir_losses = AverageMeter()  
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()

    epoch_start_time = time.time()    
    logging.info(
        'Epoch {} started at: {}'.format(epoch, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch_start_time))))  # 
    logging.info('Epoch: {} : LR = {}'.format(epoch, scheduler.get_lr()))

    for i, inputs in enumerate(train_loader):

        data_time.update(time.time() - end)
        train_step += 1
        input_dict = _parse_data(inputs)
        output_dict = model(input_dict)
        loss_dict = criterion(input_dict, output_dict, eps=epoch+1)
        loss = loss_dict["total_loss"]

        if "distance loss" in loss_dict:
            distance_losses.update(loss_dict["distance loss"].item())
        if "point_loss" in loss_dict:
            point_losses.update(loss_dict["point_loss"].item())
        if "energy_loss" in loss_dict:
            energy_losses.update(loss_dict["energy_loss"].item())
        #if "midline_loss" in loss_dict:
            #midline_losses.update(loss_dict["midline_loss"].item())
        if "cls_loss" in loss_dict:
            cls_losses.update(loss_dict["cls_loss"].item())
        if "dir_loss" in loss_dict:
            dir_losses.update(loss_dict["dir_loss"].item())

        optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        if cfg.viz and (i % cfg.viz_freq == 0 and i > 0) and epoch % 8 == 0:
            visualize_network_output(output_dict, input_dict, mode='train')

        if writer is not None and i % (cfg.display_freq * 5) == 0:
            writer.add_scalar('Train/Batch_Total_Loss', loss.item(), train_step)
            writer.add_scalar('Train/Learning_Rate', scheduler.get_lr()[0], train_step)

            for loss_name, loss_value in loss_dict.items():
                if isinstance(loss_value, torch.Tensor):
                    writer.add_scalar(f'Train/Batch_{loss_name}', loss_value.item(), train_step)

        if i % cfg.display_freq == 0:
            gc.collect()
            print_inform = "({:d} / {:d}) ".format(i, len(train_loader))
            for (k, v) in loss_dict.items():
                print_inform += " {}: {:.4f} ".format(k, v.item())
            logging.info(print_inform)

    scheduler.step()

    if writer is not None:
        writer.add_scalar('Train/Epoch_Total_Loss', losses.avg, epoch)
        writer.add_scalar('Train/Epoch_Learning_Rate', scheduler.get_lr()[0], epoch)

        if distance_losses.count > 0:
            writer.add_scalar('Train/Epoch_Distance_Loss', distance_losses.avg, epoch)
        if point_losses.count > 0:
            writer.add_scalar('Train/Epoch_Point_Loss', point_losses.avg, epoch)
        if energy_losses.count > 0:
            writer.add_scalar('Train/Epoch_Energy_Loss', energy_losses.avg, epoch)
        if midline_losses.count > 0:
            writer.add_scalar('Train/Epoch_Midline_Loss', midline_losses.avg, epoch)
        if cls_losses.count > 0:
            writer.add_scalar('Train/Epoch_Cls_Loss', cls_losses.avg, epoch)
        if dir_losses.count > 0:
            writer.add_scalar('Train/Epoch_Dir_Loss', dir_losses.avg, epoch)

    is_best_total = False
    is_best_distance = False
    is_best_point = False
    is_best_energy = False
    is_best_midline = False
    is_best_cls = False
    is_best_dir = False

    is_best = False
    if losses.avg < best_loss:
        best_loss = losses.avg
        best_epoch = epoch
        is_best_total = True
        logging.info(f'New best total loss: {best_loss:.4f} at epoch {best_epoch}')

    if distance_losses.count > 0 and distance_losses.avg < best_distance_loss:
        best_distance_loss = distance_losses.avg
        best_distance_epoch = epoch
        is_best_distance = True
        logging.info(f'New best distance loss: {best_distance_loss:.4f} at epoch {best_distance_epoch}')

    if point_losses.count > 0 and point_losses.avg < best_point_loss:
        best_point_loss = point_losses.avg
        best_point_epoch = epoch
        is_best_point = True
        logging.info(f'New best point loss: {best_point_loss:.4f} at epoch {best_point_epoch}')

    if energy_losses.count > 0 and energy_losses.avg < best_energy_loss:
        best_energy_loss = energy_losses.avg
        best_energy_epoch = epoch
        is_best_energy = True
        logging.info(f'New best energy loss: {best_energy_loss:.4f} at epoch {best_energy_epoch}')

    if midline_losses.count > 0 and midline_losses.avg < best_midline_loss:
        best_midline_loss = midline_losses.avg
        best_midline_epoch = epoch
        is_best_midline = True
        logging.info(f'New best midline loss: {best_midline_loss:.4f} at epoch {best_midline_epoch}')

    if cls_losses.count > 0 and cls_losses.avg < best_cls_loss:
        best_cls_loss = cls_losses.avg
        best_cls_epoch = epoch
        is_best_cls = True
        logging.info(f'New best classification loss: {best_cls_loss:.4f} at epoch {best_cls_epoch}')

    if dir_losses.count > 0 and dir_losses.avg < best_dir_loss:
        best_dir_loss = dir_losses.avg
        best_dir_epoch = epoch
        is_best_dir = True
        logging.info(f'New best direction loss: {best_dir_loss:.4f} at epoch {best_dir_epoch}')

    if writer is not None:
        writer.add_scalar('Best/Total_Loss', best_loss, epoch)
        writer.add_scalar('Best/Distance_Loss', best_distance_loss, epoch)
        writer.add_scalar('Best/Point_Loss', best_point_loss, epoch)
        writer.add_scalar('Best/Energy_Loss', best_energy_loss, epoch)
        writer.add_scalar('Best/Midline_Loss', best_midline_loss, epoch)
        writer.add_scalar('Best/Cls_Loss', best_cls_loss, epoch)
        writer.add_scalar('Best/Dir_Loss', best_dir_loss, epoch)

    save_model(model, epoch, scheduler.get_lr(), optimizer, scheduler,
                is_best=is_best_total, is_best_distance=is_best_distance,
                is_best_point=is_best_point, is_best_energy=is_best_energy,
                is_best_midline=is_best_midline, is_best_cls=is_best_cls, is_best_dir=is_best_dir)  

    save_model(model, epoch, scheduler.get_lr(), optimizer, scheduler, is_best=False, is_best_f1=False,
               is_best_distance=False, is_best_point=False, is_best_energy=False,
               is_best_midline=False, is_best_cls=False, is_best_dir=False)

    if cfg.exp_name in ['Uyghur']:
        val_loader = None             
        logging.info('{} No Validate'.format(epoch))   
    if val_loader is not None:
        output_dir = os.path.join(cfg.output_dir, cfg.exp_name)
        precision, recall, f1, fps = validate(model, val_loader, output_dir, epoch, dataset_type, optimizer, scheduler)

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time 
    if epoch_duration < 60: 
        logging.info('Epoch {} finished. Total time: {:.2f} seconds'.format(epoch, epoch_duration)) 
    else:
        logging.info('Epoch {} finished. Total time: {:.2f} minutes'.format(epoch, epoch_duration / 60)) 

    logging.info('=' * 80)
    logging.info('EPOCH {} TRAINING SUMMARY:'.format(epoch))
    logging.info('=' * 80) 
    logging.info('Current Losses:') 
    logging.info('  - Total Loss: {:.4f}'.format(losses.avg))
    if cls_losses.count > 0:
        logging.info('  - Classification Loss: {:.4f}'.format(cls_losses.avg))
    if distance_losses.count > 0:
        logging.info('  - Distance Loss: {:.4f}'.format(distance_losses.avg))
    if dir_losses.count > 0:
        logging.info('  - Direction Loss: {:.4f}'.format(dir_losses.avg))
    if point_losses.count > 0:
        logging.info('  - Point Loss: {:.4f}'.format(point_losses.avg))
    if energy_losses.count > 0:
        logging.info('  - Energy Loss: {:.4f}'.format(energy_losses.avg))
    if midline_losses.count > 0:
        #logging.info('  - Midline Loss: {:.4f}'.format(midline_losses.avg))

    logging.info('-' * 80) 
    logging.info('Best Losses So Far:') 
    logging.info('  - Best Total Loss: {:.4f} at epoch {}'.format(best_loss, best_epoch)) 
    logging.info('  - Best Classification Loss: {:.4f} at epoch {}'.format(best_cls_loss, best_cls_epoch)) 
    logging.info(
        '  - Best Distance Loss: {:.4f} at epoch {}'.format(best_distance_loss, best_distance_epoch)) 
    logging.info('  - Best Direction Loss: {:.4f} at epoch {}'.format(best_dir_loss, best_dir_epoch))
    logging.info('  - Best Point Loss: {:.4f} at epoch {}'.format(best_point_loss, best_point_epoch)) 
    logging.info('  - Best Energy Loss: {:.4f} at epoch {}'.format(best_energy_loss, best_energy_epoch))
    logging.info('  - Best Midline Loss: {:.4f} at epoch {}'.format(best_midline_loss, best_midline_epoch))
    logging.info('  - Best F1: at epoch {} (P: {:.4f}, R: {:.4f}, F1: {:.4f})'.format(
        best_f1_epoch, best_f1_precision, best_f1_recall, best_f1))
    logging.info('=' * 80)

def main():
    global lr, writer
    dataset_configs = {
        'Totaltext': {
            'class': TotalText,
            'data_root': 'data/total-text-mat',
            'eval_func': deal_eval_total_text,
            'transfer_func': None
        },
        'Totaltext_mid': {
            'class': TotalText,
            'data_root': 'data/total-text-mat',
            'eval_func': deal_eval_total_text,
            'transfer_func': None
        },
        'Ctw1500': {
            'class': Ctw1500Text,
            'data_root': 'data/ctw1500_eval',
            'eval_func': deal_eval_ctw1500,
            'transfer_func': None
        },
        'Ctw1500_mid': {
            'class': Ctw1500Text,
            'data_root': 'data/ctw1500_eval',
            'eval_func': deal_eval_ctw1500,
            'transfer_func': None
        },
        'Uyghur': {
            'class': Uyghur,
            'data_root': 'data/Uyghur',
            'eval_func': deal_eval_uyghur,
            'transfer_func': data_transfer_uyghur
        },
        'Uyghur_mid': {
            'class': Uyghur,
            'data_root': 'data/Uyghur',
            'eval_func': deal_eval_uyghur,
            'transfer_func': data_transfer_uyghur
        },
    }

    dataset_type = cfg.exp_name
    if dataset_type in dataset_configs:
        dataset_config = dataset_configs[dataset_type]
        trainset = dataset_config['class'](
            data_root=dataset_config['data_root'],
            ignore_list=None,
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = dataset_config['class'](
            data_root=cfg.test_root if cfg.test_root else dataset_config['data_root'],
            ignore_list=None,
            is_training=False,
            load_memory=cfg.load_memory,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        ) if dataset_config['data_root'] else None 
    else:
        logging.error(f"Unsupported dataloader name: {dataset_type}")
        return

    train_loader = data.DataLoader(
        trainset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, generator=torch.Generator(device=cfg.device)
    )

    val_loader = data.DataLoader(
        valset, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True, generator=torch.Generator(device=cfg.device)
    ) if valset is not None else None

    model = TextNet(backbone=cfg.net, is_training=True)
    model = model.to(cfg.device)
    if cfg.know:
        know_model = TextNet(backbone=cfg.knownet, is_training=False)
        load_model(know_model, cfg.know_resume)
        know_model.eval()
        know_model.requires_grad = False

    criterion = TextLoss()

    if cfg.mgpu:
        model = nn.DataParallel(model)
    if cfg.cuda:
        cudnn.benchmark = True

    lr = cfg.lr
    moment = cfg.momentum
    if cfg.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=moment)
    if cfg.exp_name == 'Synthtext':
        scheduler = FixLR(optimizer)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    if cfg.resume:
        last_epoch = load_model(model, cfg.resume, optimizer, scheduler)
        cfg.start_epoch = last_epoch + 1
        logging.info(f"Resuming from epoch {cfg.start_epoch}, LR = {optimizer.param_groups[0]['lr']:.6f}")
    logging.info('Start training CFIMEEMNet-U.')

    try:
        for epoch in range(cfg.start_epoch, cfg.max_epoch + 1):
            scheduler.step()
            train(model, train_loader, criterion, scheduler, optimizer, epoch, val_loader, dataset_type)
    finally:
        if writer is not None:
            writer.close()
            logging.info('TensorBoard writer closed.')

    logging.info('End.')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    np.random.seed(2022)
    torch.manual_seed(2022)
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    setup_logging(cfg.exp_name)
    logging_config(cfg)
    main()

