import warnings

import torch
import torch.nn.functional as F
import torch.nn as nn
from mmocr.registry import MODELS
from mmocr.models.common.losses import MaskedDiceLoss
from mmengine.model import BaseModule


@MODELS.register_module()
class IdentityHead(BaseModule):
    """The class for DBNet aux seg head.

    Args:
        in_channels (int): The number of input channels of the db head.
        with_bias (bool): Whether add bias in Conv2d layer.
        downsample_ratio (float): The downsample ratio of pred maps.
        loss (dict): Config of loss for dbnet.
        postprocessor (dict): Config of postprocessor for dbnet.
    """

    def __init__(
            self,
            downsample_ratio=1.0,
            loss_weight=1,
            reduction='mean',
            negative_ratio=3.0,
            eps=1e-6,
            bbce_loss=False,
            init_cfg=[
                dict(type='Kaiming', layer='Conv'),
                dict(type='Constant', layer='BatchNorm', val=1., bias=1e-4)
            ],
            **kwargs):
        super().__init__(
            init_cfg=init_cfg)

        assert reduction in ['mean',
                             'sum'], " reduction must in ['mean','sum']"
        self.downsample_ratio = float(downsample_ratio)
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.bbce_loss = bbce_loss
        self.dice_loss = MaskedDiceLoss(eps=eps)

        self.sigmod = nn.Sigmoid()

        self.fp16_enabled = False

    def balance_bce_loss(self, pred, gt, mask):

        positive = (gt * mask)
        negative = ((1 - gt) * mask)
        positive_count = int(positive.float().sum())
        negative_count = min(
            int(negative.float().sum()),
            int(positive_count * self.negative_ratio))

        assert gt.max() <= 1 and gt.min() >= 0
        # assert pred.max() <= 1 and pred.min() >= 0
        # gt: (N, H, W), pred: (N, 1, H, W)
        if len(gt.size()) != pred.size():
            gt = gt.unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        # loss = F.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()

        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
                positive_count + negative_count + self.eps)

        return balance_loss

    def bitmasks2tensor(self, bitmasks, target_sz):
        """Convert Bitmasks to tensor.

        Args:
            bitmasks (list[BitmapMasks]): The BitmapMasks list. Each item is
                for one img.
            target_sz (tuple(int, int)): The target tensor of size
                :math:`(H, W)`.

        Returns:
            list[Tensor]: The list of kernel tensors. Each element stands for
            one kernel level.
        """
        assert isinstance(bitmasks, list)
        assert isinstance(target_sz, tuple)

        batch_size = len(bitmasks)
        num_levels = len(bitmasks[0])

        result_tensors = []

        for level_inx in range(num_levels):
            kernel = []
            for batch_inx in range(batch_size):
                mask = torch.from_numpy(bitmasks[batch_inx].masks[level_inx])
                mask_sz = mask.shape
                pad = [
                    0, target_sz[1] - mask_sz[1], 0, target_sz[0] - mask_sz[0]
                ]
                mask = F.pad(mask, pad, mode='constant', value=0)
                kernel.append(mask)
            kernel = torch.stack(kernel)
            result_tensors.append(kernel)

        return result_tensors

    # @force_fp32(apply_to=('pred_prob',))
    def loss(self, pred_prob, gt_shrink, gt_shrink_mask, **kwargs):
        assert isinstance(gt_shrink, list)
        assert isinstance(gt_shrink_mask, list)

        # recover pred to origin input image size (gt size)
        pred_size = pred_prob.shape[2:]
        rescale_size = tuple([x * int(self.downsample_ratio) for x in pred_size]) # stage3 1/32, stage2 1/16
        pred_prob = F.interpolate(input=pred_prob, size=rescale_size, mode='bilinear', align_corners=False)
        # pred_prob = F.upsample(pred_prob, rescale_size, mode='bilinear')
        # N, 1, H, W
        feature_sz = pred_prob.size()

        keys = ['gt_shrink', 'gt_shrink_mask']
        gt = {}
        for k in keys:
            gt[k] = eval(k)
            gt[k] = [item.rescale(1.0) for item in gt[k]] # cpu operation, time-consuming if rescale isn't 1.0
            gt[k] = self.bitmasks2tensor(gt[k], feature_sz[2:])
            gt[k] = [item.to(pred_prob.device) for item in gt[k]]
        gt['gt_shrink'][0] = (gt['gt_shrink'][0] > 0).float()

        loss_prob = self._loss(pred_prob, gt['gt_shrink'][0], gt['gt_shrink_mask'][0])

        results = dict(
            loss_pix_cls=self.loss_weight * loss_prob)

        return results

    def _loss(self, pred_prob, gt, gt_mask):
        if self.bbce_loss:
            loss_prob = self.balance_bce_loss(pred_prob, gt, gt_mask)
        else:
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1, https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/dice.py
            pred_prob = F.logsigmoid(pred_prob).exp()  # binary_class
            # pred_prob = pred_prob.log_softmax(dim=1).exp() # multiclass
            loss_prob = self.dice_loss(pred_prob, gt, gt_mask)
        return loss_prob

    def forward(self, inputs):
        # inputs matching score mpa has been normalized and value is [0, 1],
        # but it divide by tau, so here inputs is logits
        return inputs
        # return self.sigmod(inputs)