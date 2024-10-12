import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

import torch
from torchvision import ops
import torch.nn.functional as F 
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def get_anc_centers(out_size):
    out_h, out_w = out_size
    anc_x = torch.arange(0, out_w) + 0.5
    anc_y = torch.arange(0, out_h) + 0.5
    
    return anc_x, anc_y

def project_bboxes(bboxes, width_scale_factor, height_scale_factor, mode='f2p'):
    assert mode in ['f2p', 'p2f']
    
    batch_size = bboxes.size(dim=0)
    proj_bboxes = bboxes.clone().reshape(batch_size, -1, 4)
    invalid_bbox_mask = (proj_bboxes == -1)
    
    if mode == 'f2p':
        # feature map to pixel image
        proj_bboxes[:, :, [0,2]] *= width_scale_factor   # x_min, x_max
        proj_bboxes[:, :, [1,3]] *= height_scale_factor  # y_min, y_max
    else:
        proj_bboxes[:, :, [0, 2]] /= width_scale_factor
        proj_bboxes[:, :, [1, 3]] /= height_scale_factor
        
    proj_bboxes.masked_fill_(invalid_bbox_mask, -1)
    proj_bboxes.resize_as_(bboxes)
    
    return proj_bboxes

def generate_proposal(anchors, offsets, device='cuda'):
    # Change format of the anchor boxes from x_min, y_min, x_max, y_max to x_center, y_center, width, height
    anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')

    anchors = anchors.to(device) 
    offsets = offsets.to(anchors.device)
    
    # Apply offsets to anchors to get proposals
    proposals = torch.zeros_like(anchors)
    proposals[:,0] = anchors[:,0] + offsets[:,0]*anchors[:,2]
    proposals[:,1] = anchors[:,1] + offsets[:,1]*anchors[:,3]
    proposals[:,2] = anchors[:,2] * torch.exp(offsets[:,2])
    proposals[:,3] = anchors[:,3] * torch.exp(offsets[:,3])
    
    # Convert format of proposals back from 'cx,cy,w,h' to 'xyxy'
    proposals = ops.box_convert(proposals, in_fmt='cxcywh', out_fmt='xyxy')
    
    return proposals

def gen_anc_base(anc_x, anc_y, anc_scales, anc_ratios, out_size):
    n_anc_boxes = 9
    anc_base = torch.zeros(1, anc_x.size(dim=0), anc_y.size(dim=0), n_anc_boxes, 4)   # Shape [1, h, w, n_anchors, 4]
    
    for ix, xc in enumerate(anc_x):
        for jx, yc in enumerate(anc_y):
            anc_boxes = torch.zeros((n_anc_boxes, 4))
            c = 0
            for i, scale in enumerate(anc_scales):
                for j, ratio in enumerate(anc_ratios):
                    w = scale * ratio
                    h = scale
                    
                    xmin = xc - w / 2
                    ymin = yc - h / 2
                    xmax = xc + w / 2
                    ymax = yc + h / 2

                    anc_boxes[c, :] = torch.Tensor([xmin, ymin, xmax, ymax])
                    c += 1

            anc_base[:, ix, jx, :] = ops.clip_boxes_to_image(anc_boxes, size=out_size)
            
    return anc_base

def get_iou_mat(batch_size, anc_boxes_all, gt_bboxes_all):
    
    # flatten anchor boxes
    anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)
    # get total anchor boxes for a single image
    tot_anc_boxes = anc_boxes_flat.size(dim=1)
    
    # create a placeholder to compute IoUs amongst the boxes
    ious_mat = torch.zeros((batch_size, tot_anc_boxes, gt_bboxes_all.size(dim=1)))

    # compute IoU of the anc boxes with the gt boxes for all the images
    for i in range(batch_size):
        gt_bboxes = gt_bboxes_all[i]
        anc_boxes = anc_boxes_flat[i]
        ious_mat[i, :] = ops.box_iou(anc_boxes, gt_bboxes)
        
    return ious_mat

def gen_anc_centers(out_size):
    out_h, out_w = out_size
    
    anc_x = torch.arange(0, out_w) + 0.5
    anc_y = torch.arange(0, out_h) + 0.5
    
    return anc_x, anc_y

def generate_proposals(anchors, offsets):
    # change format of the anchor boxes from 'xyxy' to 'cxcywh'
    anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')

    # apply offsets to anchors to create proposals
    proposals_ = torch.zeros_like(anchors)
    proposals_[:,0] = anchors[:,0] + offsets[:,0]*anchors[:,2]
    proposals_[:,1] = anchors[:,1] + offsets[:,1]*anchors[:,3]
    proposals_[:,2] = anchors[:,2] * torch.exp(offsets[:,2])
    proposals_[:,3] = anchors[:,3] * torch.exp(offsets[:,3])

    # change format of proposals back from 'cxcywh' to 'xyxy'
    proposals = ops.box_convert(proposals_, in_fmt='cxcywh', out_fmt='xyxy')

    return proposals

def get_req_anchors(anc_boxes_all, gt_bboxes_all, gt_classes_all, pos_thresh=0.7, neg_thresh=0.2):
    batch_size, w_amap, h_amap, n_anchors, _ = anc_boxes_all.shape
    N = gt_bboxes_all.shape[1]
    
    # total number of anchor box
    total_anc_boxes = n_anchors * w_amap * h_amap
    # compute iou_matrix
    anc_boxes_all = anc_boxes_all.to(gt_bboxes_all.device)
    iou_mat = get_iou_mat(batch_size, anc_boxes_all, gt_bboxes_all)
    # find max iou corresponding to gt
    max_iou_per_gt, _ = iou_mat.max(dim=1, keepdim=True)
    
    # get positive anchor boxes
    
    # condition 1: the anchor box with the max iou for every gt bbox
    positive_anc_mask = torch.logical_and(iou_mat == max_iou_per_gt, max_iou_per_gt > 0) 
    # condition 2: anchor boxes with iou above a threshold with any of the gt bboxes
    positive_anc_mask = torch.logical_or(positive_anc_mask, iou_mat > pos_thresh)
    
    positive_anc_ind_sep = torch.where(positive_anc_mask)[0] # get separate indices in the batch
    # combine all the batches and get the idxs of the +ve anchor boxes
    positive_anc_mask = positive_anc_mask.flatten(start_dim=0, end_dim=1)
    positive_anc_ind = torch.where(positive_anc_mask)[0]
    
    '''
    positive_anc_box = torch.logical_and(iou_mat == max_iou_per_gt, max_iou_per_gt > 0)
    positive_anc_box = torch.logical_or(positive_anc_box, iou_mat > pos_thresh)
    
    positive_anc_ind_sep = torch.where(positive_anc_mask)[0] # get separate indices in the batch
    # combine all the batches and get the idxs of the +ve anchor boxes
    positive_anc_mask = positive_anc_mask.flatten(start_dim=0, end_dim=1)
    positive_anc_ind = torch.where(positive_anc_mask)[0]
    '''
    
    
    # for every anchor box, get the iou and the idx of the
    # gt bbox it overlaps with the most
    max_iou_per_anc, max_iou_per_anc_ind = iou_mat.max(dim=-1)
    max_iou_per_anc = max_iou_per_anc.flatten(start_dim=0, end_dim=1)
    
    # get iou scores of the +ve anchor boxes
    GT_conf_scores = max_iou_per_anc[positive_anc_ind]
    
    # get gt classes of the +ve anchor boxes
    
    max_iou_per_anc_ind = max_iou_per_anc_ind.to(gt_bboxes_all.device)

    # expand gt classes to map against every anchor box
    gt_classes_expand = gt_classes_all.view(batch_size, 1, N).expand(batch_size, total_anc_boxes, N)
    # for every anchor box, consider only the class of the gt bbox it overlaps with the most
    GT_class = torch.gather(gt_classes_expand, -1, max_iou_per_anc_ind.unsqueeze(-1)).squeeze(-1)
    # combine all the batches and get the mapped classes of the +ve anchor boxes
    GT_class = GT_class.flatten(start_dim=0, end_dim=1)
    GT_class_pos = GT_class[positive_anc_ind]
    
    # get gt bbox coordinates of the +ve anchor boxes
    
    # expand all the gt bboxes to map against every anchor box
    gt_bboxes_expand = gt_bboxes_all.view(batch_size, 1, N, 4).expand(batch_size, total_anc_boxes, N, 4)
    # for every anchor box, consider only the coordinates of the gt bbox it overlaps with the most
    GT_bboxes = torch.gather(gt_bboxes_expand, -2, max_iou_per_anc_ind.reshape(batch_size, total_anc_boxes, 1, 1).repeat(1, 1, 1, 4))
    # combine all the batches and get the mapped gt bbox coordinates of the +ve anchor boxes
    GT_bboxes = GT_bboxes.flatten(start_dim=0, end_dim=2)
    GT_bboxes_pos = GT_bboxes[positive_anc_ind]
    
    # get coordinates of +ve anc boxes
    anc_boxes_flat = anc_boxes_all.flatten(start_dim=0, end_dim=-2) # flatten all the anchor boxes
    positive_anc_coords = anc_boxes_flat[positive_anc_ind]
    
    # calculate gt offsets
    GT_offsets = calc_gt_offsets(positive_anc_coords, GT_bboxes_pos)
    
    # get -ve anchors
    
    # condition: select the anchor boxes with max iou less than the threshold
    negative_anc_mask = (max_iou_per_anc < neg_thresh)
    negative_anc_ind = torch.where(negative_anc_mask)[0]
    # sample -ve samples to match the +ve samples
    negative_anc_ind = negative_anc_ind[torch.randint(0, negative_anc_ind.shape[0], (positive_anc_ind.shape[0],))]
    negative_anc_coords = anc_boxes_flat[negative_anc_ind]
    
    return positive_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class_pos, positive_anc_coords, negative_anc_coords, positive_anc_ind_sep

def calc_gt_offsets(pos_anc_coords, gt_bbox_mapping):
    pos_anc_coords = ops.box_convert(pos_anc_coords, in_fmt='xyxy', out_fmt='cxcywh')
    gt_bbox_mapping = ops.box_convert(gt_bbox_mapping, in_fmt='xyxy', out_fmt='cxcywh')

    gt_cx, gt_cy, gt_w, gt_h = gt_bbox_mapping[:, 0], gt_bbox_mapping[:, 1], gt_bbox_mapping[:, 2], gt_bbox_mapping[:, 3]
    anc_cx, anc_cy, anc_w, anc_h = pos_anc_coords[:, 0], pos_anc_coords[:, 1], pos_anc_coords[:, 2], pos_anc_coords[:, 3]

    tx_ = (gt_cx - anc_cx)/anc_w
    ty_ = (gt_cy - anc_cy)/anc_h
    tw_ = torch.log(gt_w / anc_w)
    th_ = torch.log(gt_h / anc_h)

    return torch.stack([tx_, ty_, tw_, th_], dim=-1)
