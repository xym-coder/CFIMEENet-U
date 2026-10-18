import os
import time
import cv2
import numpy as np
import json
from shapely.geometry import *
import torch
import subprocess
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from dataloader import  Ctw1500Text, TotalText, Uyghur
from model.textnet import TextNet
from config.config import config as cfg, update_config, logging_config
from config.commandline_parameters import BaseOptions
from tool_functions.augmentation import BaseTransform
from tool_functions.visualize import visualize_detection, visualize_gt
from tool_functions.base_tool_functions import to_device, mkdirs,rescale_result, get_cosine_map
from tool_functions.eval import deal_eval_total_text, deal_eval_ctw1500, deal_eval_uyghur, data_transfer_uyghur
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

def osmkdir(out_dir):
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

def write_to_file(contours, file_path):
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 0], cont[:, 1]], 1)
            if cv2.contourArea(cont) <= 0:
                continue
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')

def inference(model, test_loader, output_dir):
    total_time = 0.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    art_results = dict()

    for i, (image, meta) in enumerate(test_loader):
        input_dict = dict()
        idx = 0
        H, W = meta['Height'][idx].item(), meta['Width'][idx].item()
        image = image.to(device,non_blocking=True)
        input_dict['img'] = image
        # get detection result
        start = time.time()

        with torch.no_grad():
            output_dict = model(input_dict)
        torch.cuda.synchronize()
        end = time.time()
        if i > 0:
            total_time += end - start
            fps = (i + 1) / total_time
        else:
            fps = 0.0

        print('detect {} / {} images: {}. ({:.2f} fps)'.format(i + 1, len(test_loader), meta['image_id'][idx], fps), end = '\r', flush = True)

        img_show = image[idx].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        if cfg.viz:
            gt_contour = []
            label_tag = meta['label_tag'][idx].int().cpu().numpy()
            for annot, n_annot in zip(meta['annotation'][idx], meta['n_annotation'][idx]):
                if n_annot.item() > 0:
                    gt_contour.append(annot[:n_annot].int().cpu().numpy())
            gt_vis = visualize_gt(img_show, gt_contour, label_tag)
            show_boundary, heat_map = visualize_detection(img_show, output_dict, meta=meta)
            show_map = np.concatenate([heat_map, gt_vis], axis=1)
            show_map = cv2.resize(show_map, (320 * 3, 320))
            show_boundary = cv2.resize(show_boundary, (show_map.shape[1], show_boundary.shape[0]))
            im_vis = np.concatenate([show_map, show_boundary], axis=0)
            path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name), meta['image_id'][idx].split(".")[0]+".jpg")
            cv2.imwrite(path, im_vis)
            # print(path)

        contours = output_dict["py_preds"][-1].int().cpu().numpy()
        img_show, contours = rescale_result(img_show, contours, H, W)
        # empty GPU  cache
        torch.cuda.empty_cache()

        if cfg.exp_name == "Totaltext" or cfg.exp_name == "Totaltext_mid":
            fname = "res_" + meta['image_id'][idx].replace('jpg', 'txt')
            contours = deal_eval_total_text(contours)
            write_to_file(contours, os.path.join(output_dir, fname))
        elif cfg.exp_name == "Ctw1500" or cfg.exp_name == "Ctw1500_mid":
            fname = "res_" + meta['image_id'][idx].split(".")[0]+".txt"
            deal_eval_ctw1500(contours, os.path.join(output_dir, fname))
        elif cfg.exp_name == "Uyghur" or cfg.exp_name == "Uyghur_mid":
            fname = "res_" + meta['image_id'][idx].split(".")[0]+".txt"
            data_transfer_uyghur(contours, os.path.join(output_dir, fname))
        else:
            tmp = np.array(contours).astype(np.int32)
            if tmp.size == 0:
                print(f"Skipping empty contours for {meta['image_id'][idx]}")
                fname = meta['image_id'][idx].replace('jpg', 'txt')
                with open(os.path.join(output_dir, fname), 'w') as f:
                    pass
                continue
            tmp = np.expand_dims(tmp, axis=2)
            # print(tmp.shape)
            fname = meta['image_id'][idx].replace('jpg', 'txt')
            write_to_file(contours, os.path.join(output_dir, fname))

def main(vis_dir_path):
    osmkdir(vis_dir_path)
    if cfg.exp_name == "Totaltext" or cfg.exp_name == "Totaltext_mid":
        testset = TotalText(
            data_root='data/total-text-mat',
            ignore_list=None,
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == "Ctw1500" or cfg.exp_name == 'Ctw1500_mid':
        testset = Ctw1500Text(
            data_root='data/ctw1500',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == "Uyghur" or cfg.exp_name == 'Uyghur_mid':
        testset = Uyghur(
            data_root='data/Uyghur',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    else:
        print("{} is not justify".format(cfg.exp_name))

    cudnn.benchmark = False
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    model = TextNet(is_training=False, backbone=cfg.net)
    model_path = os.path.join(cfg.save_dir, cfg.exp_name,
                              'MixNet_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))

    model.load_model(model_path)
    model.to(cfg.device)
    model.eval()
    with torch.no_grad():
        print('Start testing MixNet.')
        output_dir = os.path.join(cfg.output_dir, cfg.exp_name)
        inference(model, test_loader, output_dir)

    print("{} eval finished.".format(cfg.exp_name))

if __name__ == "__main__":

    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    # print_config(cfg)

    vis_dir = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name))
    if not os.path.exists(vis_dir):
        mkdirs(vis_dir)
    main(vis_dir)
