#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""RF-DETR detector integration for DeepLabCut"""
from __future__ import annotations

import logging
from pathlib import Path
import os

import torch

from deeplabcut.pose_estimation_pytorch.models.detectors.base import (
    DETECTORS,
    BaseDetector,
)

logger = logging.getLogger(__name__)

def download_pretrain_weights(pretrain_weights_name: str, pretrain_weights_dir: str, redownload=False):
    from rfdetr.main import download_file, HOSTED_MODELS
    if pretrain_weights_name in HOSTED_MODELS:
        if redownload or not os.path.exists(os.path.join(pretrain_weights_dir, pretrain_weights_name)):
            logger.info(
                f"Downloading pretrained weights for {pretrain_weights_name}"
            )
            download_file(
                HOSTED_MODELS[pretrain_weights_name],
                os.path.join(pretrain_weights_dir, pretrain_weights_name),
            )
    else:
        raise ValueError(f"Pretrained weights '{pretrain_weights_name}' not found in hosted models: {list(HOSTED_MODELS.keys())}")


@DETECTORS.register_module
class RFDETR(BaseDetector):
    """RF-DETR (Roboflow Detection Transformer) detector for DeepLabCut
    
    RF-DETR is a state-of-the-art object detection model based on transformers.
    This class wraps the RF-DETR model to work within DeepLabCut's training
    and inference framework.
    
    Paper: RF-DETR: Towards Real-Time End-to-End Object Detection
    
    Args:
        model_size: Size of the RF-DETR model ('nano', 'small', 'medium', 'base', 'large').
        pretrained: Whether to load COCO pretrained weights.
        freeze_bn_stats: Whether to freeze batch norm statistics.
        freeze_bn_weights: Whether to freeze batch norm weights.
        box_score_thresh: Minimum confidence threshold for detection.
        resolution: Input resolution (must be divisible by 56).
        num_classes: Number of detection classes (default: 2, including background).
        pretrain_weights_path: Path to custom pretrained weights.
    """

    def __init__(
        self,
        model_size: str = "medium",
        pretrained: bool = True,
        freeze_bn_stats: bool = False,
        freeze_bn_weights: bool = False,
        num_classes: int = 2,
        weights_dir_path: str | None = None,
    ) -> None:
        super().__init__(
            freeze_bn_stats=freeze_bn_stats,
            freeze_bn_weights=freeze_bn_weights,
            pretrained=pretrained,
        )

        from rfdetr.main import Model
        from rfdetr.models.lwdetr import build_criterion_and_postprocessors
        from rfdetr import RFDETRBase
        from rfdetr.datasets.transforms import Normalize
        from rfdetr.config import (
            RFDETRNanoConfig,
            RFDETRSmallConfig, 
            RFDETRMediumConfig,
            RFDETRBaseConfig,
            RFDETRLargeConfig,
        )
        variant_config_map = {
            "nano": RFDETRNanoConfig,
            "small": RFDETRSmallConfig,
            "medium": RFDETRMediumConfig,
            "base": RFDETRBaseConfig,
            "large": RFDETRLargeConfig,
        }

        self.means = RFDETRBase.means
        self.stds = RFDETRBase.stds
        self.normalize = Normalize(mean=self.means, std=self.stds)
        self.model_size = model_size
        self.num_classes = num_classes
        
        weights_filename = f"rf-detr-{model_size}.pth"
        if weights_dir_path is None:
            weights_dir_path = Path(__file__).parent / "pretrained_weights"

        if pretrained:
            weights_dir_path.mkdir(exist_ok=True, parents=False)
            download_pretrain_weights(weights_filename, str(weights_dir_path), redownload=False)
        
        
        config_class = variant_config_map[model_size]
        model_config = config_class(
            num_classes=num_classes - 1,
            pretrain_weights=str(weights_dir_path / weights_filename) if pretrained else None,
        )
        self.model = Model(**model_config.dict())
        self.model.reinitialize_detection_head(self.model.args.num_classes)
        self.criterion, self.postprocess = build_criterion_and_postprocessors(self.model.args)
    
    def parameters(self, recurse = True):
        return self.model.model.parameters(recurse)
        
    def eval(self) -> None:
        """Set the module in evaluation mode"""
        super().eval()
        self.model.model.eval()
        self.criterion.eval()
    
    def train(self, mode: bool = True) -> None:
        """Set the module in training or evaluation mode"""
        super().train(mode)
        self.model.model.train(mode)
        self.criterion.train(mode)

    def get_target(self, labels: dict) -> list[dict]:
        """Returns target in a format RF-DETR can handle
        
        Args:
            labels: dict of annotations, must contain the keys:
                boxes: Tensor of shape (B, N, 4) in (x, y, w, h) format, where B is batch 
                    size and N is number of boxes per image
                labels: Tensor of shape (B, N) with 1-indexed class labels
                
                Optional keys (preserved if present):
                    area: Tensor of shape (B, N) containing bounding box areas
                    is_crowd: Tensor of shape (B, N) indicating crowd annotations
                    keypoints: Tensor of shape (B, N_ind, N_kp, 3) with keypoint data
                    individual_ids: Tensor of shape (B, N) with individual IDs
        
        Returns:
            res: list of B dictionaries, each representing target information for a single
                image in the batch. Each dictionary contains:
                    'boxes': Tensor of shape (N, 4) in (x, y, w, h) format (float32)
                    'labels': Tensor of shape (N,) with 0-indexed class labels (int64)
                    
                Note: Boxes will be converted to (x1, y1, x2, y2) format and normalized
                by the RF-DETR Normalize transform during forward pass.
        """
        
        processed_targets = []
        for i, boxes in enumerate(labels["boxes"]):

            processed_target = {}
            
            # Filter out invalid boxes (where width or height <= 0)
            mask = (boxes[:, 2] > 0.0) & (boxes[:, 3] > 0.0)
            boxes = boxes[mask]
            
            # bbox format conversion (x, y, w, h) -> (x1, y1, x2, y2)
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            
            processed_target['boxes'] = boxes.float()
            
            # Get labels for this image
            labels_raw = labels['labels'][i]
            
            # Apply the same mask to filter labels and convert to 0-indexed
            labels_raw = labels_raw[mask]
            processed_target['labels'] = (labels_raw - 1).long()
            processed_target['area'] = labels['area'][i][mask]
            processed_target['is_crowd'] = labels['is_crowd'][i][mask]
            
            processed_targets.append(processed_target)
        
        return processed_targets

    def forward(
        self,
        x: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        """Forward pass of the RF-DETR detector
        
        Args:
            x: Input images of shape (B, C, H, W)
            targets: List of target dictionaries for training, each containing:
                - 'boxes': Tensor of shape (N, 4) in xyxy format
                - 'labels': Tensor of shape (N,) with class labels
                
        Returns:
            losses: Dictionary of loss values (only during training)
            detections: List of detection dictionaries, one per image, containing:
                - 'boxes': Tensor of shape (M, 4) in xyxy format
                - 'scores': Tensor of shape (M,)
                - 'labels': Tensor of shape (M,)
        """
        import torchvision.transforms.functional as F
        
        # Process images and targets using RF-DETR's Normalize transform
        h, w = x.shape[2], x.shape[3]
        orig_sizes = [(h, w)] * x.shape[0]
        
        processed_images = []
        rfdetr_targets = []
        
        if self.training and targets is not None:
            for i, img in enumerate(x):
                target = targets[i]
                
                # Use RF-DETR's Normalize - it handles both image normalization and box conversion
                img_norm, target_norm = self.normalize(img.float(), target)
                
                # Add size fields required by RF-DETR
                target_norm['orig_size'] = torch.tensor([h, w], device=x.device)
                target_norm['size'] = torch.tensor([h, w], device=x.device)
                
                # Preserve optional fields
                if 'image_id' in target:
                    target_norm['image_id'] = target['image_id']
                
                rfdetr_targets.append(target_norm)
                
                # Resize to model resolution (same as predict method)
                img_resized = F.resize(img_norm, (self.model.resolution, self.model.resolution))
                processed_images.append(img_resized)
        else:
            # Inference mode - just normalize and resize images
            for img in x:
                img_norm, _ = self.normalize(img.float(), None)
                img_resized = F.resize(img_norm, (self.model.resolution, self.model.resolution))
                processed_images.append(img_resized)
        
        batch_tensor = torch.stack(processed_images)
        
        if self.training and targets is not None:
            predictions = self.model.model(batch_tensor)
            loss_dict = self.criterion(predictions, rfdetr_targets)
            weight_dict = self.criterion.weight_dict
            
            losses = {
                k: loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            }
            
            with torch.no_grad():
                target_sizes = torch.tensor(orig_sizes, device=x.device)
                results = self.postprocess(predictions, target_sizes)
        else:
            # Inference mode
            with torch.no_grad():
                predictions = self.model.model(batch_tensor)
                target_sizes = torch.tensor(orig_sizes, device=x.device)
                results = self.postprocess(predictions, target_sizes)
            losses = {}
        
        # Convert labels back to 1-indexed for DLC compatibility
        # (RF-DETR uses 0-indexed, DLC uses 1-indexed)
        detections = [
            {
                'boxes': result['boxes'],
                'scores': result['scores'],
                'labels': result['labels'] + 1,
            }
            for result in results
        ]
        
        return losses, detections


    def to(self, *args, **kwargs):
        """Move module to device"""
        self = super().to(*args, **kwargs)
        self.model.model = self.model.model.to(*args, **kwargs)
        self.criterion = self.criterion.to(*args, **kwargs)
        return self

    def state_dict(self, *args, **kwargs):
        """Get state dict of the RF-DETR model"""
        return self.model.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.model.load_state_dict(state_dict, strict=strict)