import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import cv2
from config.config import config as cfg
from model.backbone_feature_integration.feature_fusion import MFF
from model.backbone_feature_integration.Transformer import Transformer
from model.textnet import get_node_feature
from tool_functions.base_tool_functions import get_sample_point
from create_modules.CGS import get_midline_from_proposal_skeleton

class midlinePredictor(nn.Module):
    def __init__(self, seg_channel):
        super(midlinePredictor, self).__init__()
        self.seg_channel = seg_channel
        self.clip_dis = 100
        self.midline_preds = nn.ModuleList()
        self.contour_preds = nn.ModuleList()
        self.iter = 3

        for i in range(self.iter):

            self.midline_preds.append(
                Transformer(
                    seg_channel, 128, num_heads=8, 
                    dim_feedforward=1024, drop_rate=0.0, 
                    if_resi=True, block_nums=3, pred_num=2, batch_first=False)
            )

            self.contour_preds.append(
                Transformer(
                    seg_channel, 128, num_heads=8, 
                    dim_feedforward=1024, drop_rate=0.0, 
                    if_resi=True, block_nums=3, pred_num=2, batch_first=False)
            )
        if not self.training:
            self.iter = 1

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_boundary_proposal(self, input=None):
        inds = torch.where(input['ignore_tags'] > 0)
        init_polys = input['proposal_points'][inds]
        midlines = input['midline_points'][inds]
        # return init_polys, inds, None
        return init_polys, inds, None, midlines

    def get_boundary_proposal_eval(self, input=None, seg_preds=None):
        cls_preds = seg_preds[:, 0, :, :].detach().cpu().numpy()
        dis_preds = seg_preds[:, 1, :, :].detach().cpu().numpy()
        inds = []
        init_polys = []
        confidences = []
        midlines = []
        for bid, dis_pred in enumerate(dis_preds):

            dis_mask = dis_pred > cfg.dis_threshold

            ret, labels = cv2.connectedComponents(dis_mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_16U)

            for idx in range(1, ret):
                text_mask = labels == idx

                confidence = round(cls_preds[bid][text_mask].mean(), 3)

                if np.sum(text_mask) < 50 / (cfg.scale * cfg.scale) or confidence < cfg.cls_threshold:
                    continue
                confidences.append(confidence)
                inds.append([bid, 0])

                poly = get_sample_point(text_mask, cfg.num_points,
                                        cfg.approx_factor, scales=np.array([cfg.scale, cfg.scale]))
                init_polys.append(poly)

                midline = get_midline_from_proposal_skeleton(poly, text_mask, cfg.num_points // 2)
                midlines.append(midline)

        if len(inds) > 0:
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device, non_blocking=True)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
            midlines = torch.from_numpy(np.array(midlines)).to(input["img"].device, non_blocking=True).float()
        else
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
            inds = torch.from_numpy(np.array(inds)).to(input["img"].device, non_blocking=True)
            midlines = torch.from_numpy(np.array(midlines)).to(input["img"].device, non_blocking=True).float()

        return init_polys, inds, confidences, midlines

    def forward(self, embed_feature, input=None, seg_preds=None, switch="gt"):

        if self.training:

            init_polys, inds, confidences, midlines = self.get_boundary_proposal(input=input)
        else:

            init_polys, inds, confidences, midlines = self.get_boundary_proposal_eval(input=input, seg_preds=seg_preds)
            if init_polys.shape[0] == 0:

                return [init_polys, init_polys], inds, confidences, None

        if len(init_polys) == 0:
            py_preds = torch.zeros_like(init_polys)

        h, w = embed_feature.shape[2:4]

        contours = [init_polys]
        if self.training:
            midlines = [midlines]
        else:

            init_mid_feat = get_node_feature(embed_feature, midlines, inds[0], h, w)
            midline = midlines + torch.clamp(self.midline_preds[0](init_mid_feat).permute(0, 2, 1), -self.clip_dis, self.clip_dis)
            midlines = [midline]

        for i in range(self.iter):

            mid_feat = get_node_feature(embed_feature, midlines[i], inds[0], h, w)

            midline = midlines[i] + torch.clamp(self.midline_preds[i](mid_feat).permute(0, 2, 1), -self.clip_dis, self.clip_dis)
            midlines.append(midline)

            node_feat = get_node_feature(embed_feature, contours[i], inds[0], h, w)

            node_feat = torch.cat((node_feat, mid_feat), dim=2)

            new_contour = contours[i] + torch.clamp(self.contour_preds[i](node_feat).permute(0, 2, 1), -self.clip_dis, self.clip_dis)[:, :cfg.num_points]
            contours.append(new_contour)

        return contours, inds, confidences, midlines[1:]

  
    