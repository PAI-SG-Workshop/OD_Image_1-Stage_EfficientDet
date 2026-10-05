"""
EfficientDet-D0 builder (Image task; KITTI complex / yolo3d track: backbone + BiFPN with the YOLOv3 layers).

Paper: Tan, Pang, Le, "EfficientDet: Scalable and Efficient Object Detection" (CVPR 2020, arXiv 1911.09070).
Code : official google/automl (TensorFlow); approved official-equivalent source (2026-10-04): rwightman
       efficientdet-pytorch (pip package effdet 0.4.1, reproduces the official TF weights), used directly:
       effdet.EfficientDet (tf_efficientnet_b0 backbone via timm, BiFPN x3 with fast normalised fusion, 64 ch,
       class / box nets x3 with separable convs and per-level BN), effdet.anchors.Anchors (3 scales x 3 ratios,
       anchor_scale 4, P3-P7), AnchorLabeler (IoU 0.5), effdet.loss.DetectionLoss (focal alpha 0.25 gamma 1.5,
       Huber delta 0.1, box_loss_weight 50, / (#pos + 1)), decode_box_outputs.
       Requirements therefore include effdet (and with it timm).

effdet works in YXYX boxes and 1-based classes; the wrapper converts the framework's XYXY / 0-based targets.
Raw contract: sigmoid class probabilities, objectness = 1.

[Unified] no ImageNet backbone weights (pretrained_backbone=False); common init (nn.Conv2d incl. timm Conv2dSame,
BN; class-net prior bias kept); no mean / std; 608 x 608 instead of 512 x 512 (feature maps 76/38/19/10/5);
common augmentation instead of scale jitter; Adam 1e-4 instead of SGD 0.08 cosine + EMA; common post-processing
(instead of top-5000 + NMS 0.5, max 100). PreciseBN (BN everywhere, micro batch 2).
"""

import torch
import torch.nn as nn
from effdet import EfficientDet as EffDetNet
from effdet import get_efficientdet_config
from effdet.anchors import AnchorLabeler, Anchors, decode_box_outputs
from effdet.loss import DetectionLoss

from detectors.anchor_free.common import HeadStats, split_targets, to_raw


class EfficientDet(nn.Module):
    def __init__(self, num_classes, name="tf_efficientdet_d0", image_size=608):
        super().__init__()
        config = get_efficientdet_config(name)
        config.num_classes = num_classes
        config.image_size = (image_size, image_size)
        self.config = config
        self.num_classes = num_classes
        self.model = EffDetNet(config, pretrained_backbone=False)
        self.anchors = Anchors.from_config(config)
        self.anchor_labeler = AnchorLabeler(self.anchors, num_classes, match_threshold=0.5)
        self.loss_fn = DetectionLoss(config)
        self.precise_bn = True                 # [Unified] common/precise_bn.py
        self.precise_bn_images = 1600
        self.precise_bn_batch_size = 16
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def _flatten(self, outs, k):
        """effdet per-level [B, A*k, H, W] -> [B, sum(H*W*A), k] (same order as Anchors.boxes)."""
        B = outs[0].shape[0]
        return torch.cat([o.permute(0, 2, 3, 1).reshape(B, -1, k) for o in outs], 1)

    def forward(self, x, targets=None):
        assert tuple(x.shape[-2:]) == tuple(self.config.image_size), "EfficientDet anchors are built for image_size"
        cls_outputs, box_outputs = self.model(x)
        cls = self._flatten(cls_outputs, self.num_classes)
        box = self._flatten(box_outputs, 4)
        B, R = box.shape[:2]
        anchors = self.anchors.boxes.to(x.device)                               # [R, 4] yxyx
        boxes = decode_box_outputs(box.reshape(-1, 4).float(), anchors.repeat(B, 1), output_xyxy=True).view(B, R, 4)
        raw = to_raw(boxes, cls.sigmoid())                                      # objectness = 1
        if targets is None:
            return raw
        tgt = split_targets(targets, B, x.shape[2])
        gt_boxes = [t["boxes"][:, [1, 0, 3, 2]].float() for t in tgt]           # xyxy -> yxyx
        gt_classes = [(t["labels"] + 1).float() for t in tgt]                    # 1-based
        cls_targets, box_targets, num_positives = self.anchor_labeler.batch_label_anchors(gt_boxes, gt_classes)
        loss, class_loss, box_loss = self.loss_fn(cls_outputs, box_outputs, cls_targets, box_targets, num_positives)
        loss_dict = {"class_loss": class_loss, "box_loss": box_loss, "losses": loss,
                     "num_pos": num_positives.float().sum() / B}
        ct = torch.cat([c.reshape(B, -1) for c in cls_targets], 1).view(-1)      # -1 bg, -2 ignore, >= 0 class
        pos = ct >= 0
        cls_flat = cls.detach().reshape(-1, self.num_classes)
        self.stats.update(loss_dict, cls_flat.sigmoid().max(-1).values, cls_flat, pos, ct[pos])
        return loss, raw

    def load_darknet_weights(self, path):
        raise NotImplementedError("EfficientDet has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    return EfficientDet(num_classes)
