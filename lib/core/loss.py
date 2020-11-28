# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        w, h = output.size(2), output.size(3)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        num_joints_real = target_weight.size(1)
        if num_joints != num_joints_real:
            assert num_joints == num_joints_real * 2, "some thing unexpect hapen."

        for idx in range(num_joints):
            weight = 1 if idx < num_joints_real else 1
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx%num_joints_real]),
                    heatmap_gt.mul(target_weight[:, idx%num_joints_real])
                ) * weight
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt) * weight

        return loss / num_joints
