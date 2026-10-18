import torch
import numpy as np
import cv2
import os
import math
from config.config import config as cfg
import matplotlib
import pylab as plt
from matplotlib import cm
import torch.nn.functional as F

matplotlib.use('agg')

def visualize_network_output(output_dict, input_dict, mode='train'):

    vis_dir = os.path.join(cfg.vis_dir, cfg.exp_name + '_' + mode)

    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)

    fy_preds = F.interpolate(
        output_dict["fy_preds"],
        scale_factor=cfg.scale,
        mode='bilinear'
    )

    fy_preds = fy_preds.data.cpu().numpy()

    py_preds = output_dict["py_preds"][1:]
    init_polys = output_dict["py_preds"][0]

    inds = output_dict["inds"]

    if cfg.mid == True:
        midline = output_dict["midline"]

    image = input_dict['img']

    tr_mask = input_dict['tr_mask'].data.cpu().numpy() > 0

    distance_field = input_dict['distance_field'].data.cpu().numpy()
    direction_field = input_dict['direction_field']

    weight_matrix = input_dict['weight_matrix']

    gt_tags = input_dict['gt_points'].cpu().numpy()

    ignore_tags = input_dict['ignore_tags'].cpu().numpy()

    b, c, _, _ = fy_preds.shape

    for i in range(b):

        fig = plt.figure(figsize=(15, 9))

        mask_pred = fy_preds[i, 0, :, :]

        distance_pred = fy_preds[i, 1, :, :]

        norm_pred = np.sqrt(
            fy_preds[i, 2, :, :] ** 2 +
            fy_preds[i, 3, :, :] ** 2
        )

        angle_pred = 180 / math.pi * np.arctan2(
            fy_preds[i, 2, :, :],
            fy_preds[i, 3, :, :] + 0.00001
        )

        ax1 = fig.add_subplot(341)
        ax1.set_title('mask_pred')
        ax1.imshow(mask_pred, cmap=cm.jet)

        ax2 = fig.add_subplot(342)
        ax2.set_title('distance_pred')
        ax2.imshow(distance_pred, cmap=cm.jet)

        ax3 = fig.add_subplot(343)
        ax3.set_title('norm_pred')
        ax3.imshow(norm_pred, cmap=cm.jet)

        ax4 = fig.add_subplot(354)
        ax4.set_title('angle_pred (jet)')
        ax4.imshow(angle_pred, cmap=cm.jet, vmin=-180, vmax=180)

        ax5 = fig.add_subplot(355)
        ax5.set_title('angle_pred (hsv)')
        ax5.imshow(angle_pred, cmap=cm.hsv, vmin=-180, vmax=180)

        mask_gt = tr_mask[i]
        # GT distance field
        distance_gt = distance_field[i]
        # GT direction field
        gt_flux = direction_field[i].cpu().numpy()

        norm_gt = np.sqrt(
            gt_flux[0, :, :] ** 2 +
            gt_flux[1, :, :] ** 2
        )

        angle_gt = 180 / math.pi * np.arctan2(
            gt_flux[0, :, :],
            gt_flux[1, :, :] + 0.00001
        )
        # 显示 GT mask
        ax11 = fig.add_subplot(356)
        ax11.set_title('mask_gt')
        ax11.imshow(mask_gt, cmap=cm.jet)

        ax22 = fig.add_subplot(357)
        ax22.set_title('distance_gt')
        ax22.imshow(distance_gt, cmap=cm.jet)

        ax33 = fig.add_subplot(358)
        ax33.set_title('norm_gt')
        ax33.imshow(norm_gt, cmap=cm.jet)

        ax44 = fig.add_subplot(359)
        ax44.set_title('angle_gt (jet)')
        ax44.imshow(angle_gt, cmap=cm.jet, vmin=-180, vmax=180)

        ax55 = fig.add_subplot(3, 5, 10)
        ax55.set_title('angle_gt (hsv)')
        ax55.imshow(angle_gt, cmap=cm.hsv, vmin=-180, vmax=180)

        img_show = image[i].permute(1, 2, 0).cpu().numpy()

        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        img_show = np.ascontiguousarray(img_show[:, :, ::-1])
        shows = []

        gt = gt_tags[i]

        gt_idx = np.where(ignore_tags[i] > 0)
        gt_py = gt[gt_idx[0], :, :]

        index = torch.where(inds[0] == i)[0]

        init_py = init_polys[index].detach().cpu().numpy()

        image_show = img_show.copy()
        cv2.drawContours(
            image_show,
            init_py.astype(np.int32),
            -1,
            (255, 0, 0),
            2
        )
        cv2.drawContours(
            image_show,
            gt_py.astype(np.int32),
            -1,
            (0, 255, 255),
            2
        )
        shows.append(image_show)

        for py in py_preds:
            contours = py[index].detach().cpu().numpy()

            image_show = img_show.copy()

            cv2.drawContours(
                image_show,
                init_py.astype(np.int32),
                -1,
                (0, 125, 125),
                2
            )

            cv2.drawContours(
                image_show,
                gt_py.astype(np.int32),
                -1,
                (255, 125, 0),
                2
            )

            cv2.drawContours(
                image_show,
                contours.astype(np.int32),
                -1,
                (0, 255, 125),
                2
            )
            if cfg.mid == True:
                if midline is not None:
                    cv2.polylines(
                        image_show,
                        midline.detach().cpu().numpy().astype(np.int32),
                        False,
                        (125, 255, 0),
                        2
                    )
            shows.append(image_show)

        for idx, im_show in enumerate(shows):
            axb = fig.add_subplot(3, 5, 11 + idx)
            axb.imshow(im_show)
        path = os.path.join(vis_dir, '{}.png'.format(i))
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

def visualize_gt(image, contours, label_tag):
    image_show = image.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])
    image_show = cv2.polylines(
        image_show,
        [contours[i] for i, tag in enumerate(label_tag) if tag > 0],
        True,
        (0, 0, 255),
        3
    )
    image_show = cv2.polylines(
        image_show,
        [contours[i] for i, tag in enumerate(label_tag) if tag < 0],
        True,
        (0, 255, 0),
        3
    )
    show_gt = cv2.resize(image_show, (320, 320))

    return show_gt

def heatmap(im_gray):
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(255 - im_gray)
    Hmap = np.delete(rgba_img, 3, 2)

    return Hmap

def visualize_detection(image, output_dict, meta=None):
    image_show = image.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])
    cls_preds = F.interpolate(
        output_dict["fy_preds"],
        scale_factor=cfg.scale,
        mode='bilinear'
    )
    cls_preds = cls_preds[0].data.cpu().numpy()
    py_preds = output_dict["py_preds"][1:]
    init_polys = output_dict["py_preds"][0]
    shows = []

    if cfg.mid:
        midline = output_dict["midline"]

    init_py = init_polys.data.cpu().numpy()

    path = os.path.join(
        cfg.vis_dir,
        '{}_test'.format(cfg.exp_name),
        meta['image_id'][0].split(".")[0] + "_init.png"
    )

    im_show0 = image_show.copy()
    for i, bpts in enumerate(init_py.astype(np.int32)):

        cv2.drawContours(
            im_show0,
            [bpts.astype(np.int32)],
            -1,
            (255, 0, 255),
            4
        )

        for j, pp in enumerate(bpts):
            if j == 0:
                cv2.circle(
                    im_show0,
                    (int(pp[0]), int(pp[1])),
                    3,
                    (125, 125, 255),
                    -1
                )
            elif j == 1:
                cv2.circle(
                    im_show0,
                    (int(pp[0]), int(pp[1])),
                    3,
                    (125, 255, 125),
                    -1
                )
            else:
                cv2.circle(
                    im_show0,
                    (int(pp[0]), int(pp[1])),
                    3,
                    (255, 125, 125),
                    -1
                )

    cv2.imwrite(path, im_show0)

    for idx, py in enumerate(py_preds):

        im_show = im_show0.copy()
        contours = py.data.cpu().numpy()
        cv2.drawContours(
            im_show,
            contours.astype(np.int32),
            -1,
            (0, 255, 255),
            4
        )
        for ppts in contours:
            for j, pp in enumerate(ppts):
                if j == 0:
                    cv2.circle(
                        im_show,
                        (int(pp[0]), int(pp[1])),
                        3,
                        (125, 125, 255),
                        -1
                    )
                elif j == 1:
                    cv2.circle(
                        im_show,
                        (int(pp[0]), int(pp[1])),
                        3,
                        (125, 255, 125),
                        -1
                    )
                else:
                    cv2.circle(
                        im_show,
                        (int(pp[0]), int(pp[1])),
                        3,
                        (255, 125, 125),
                        -1
                    )

        if cfg.mid and midline is not None:

            if idx < len(midline):
                current_midline = midline[idx]

                if isinstance(current_midline, torch.Tensor):
                    current_midline = (
                        current_midline
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int32)
                    )
                elif isinstance(current_midline, list):
                    current_midline = np.array(current_midline).astype(np.int32)

                if current_midline.ndim == 3:
                    for line_idx, ml in enumerate(current_midline):
                        if ml.shape[-1] == 2 and len(ml) > 1:
                            cv2.polylines(
                                im_show,
                                [ml],
                                False,
                                (0, 255, 0),
                                4
                            )

                elif current_midline.ndim == 2 and current_midline.shape[-1] == 2:
                    cv2.polylines(
                        im_show,
                        [current_midline],
                        False,
                        (0, 255, 0),
                        4
                    )

        path = os.path.join(
            cfg.vis_dir,
            '{}_test'.format(cfg.exp_name),
            meta['image_id'][0].split(".")[0] + "_{}iter.png".format(idx)
        )
        cv2.imwrite(path, im_show)

        shows.append(im_show)

    show_img = np.concatenate(shows, axis=1)
    show_boundary = cv2.resize(
        show_img,
        (320 * len(py_preds), 320)
    )
    cls_pred = heatmap(
        np.array(cls_preds[0] * 255, dtype=np.uint8)
    )
    dis_pred = heatmap(
        np.array(cls_preds[1] * 255, dtype=np.uint8)
    )

    heat_map = np.concatenate(
        [cls_pred * 255, dis_pred * 255],
        axis=1
    )

    heat_map = cv2.resize(
        heat_map,
        (320 * 2, 320)
    )
    return show_boundary, heat_map