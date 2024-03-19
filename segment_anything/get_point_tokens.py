import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple

from .modeling import Sam
from .predictor import SamPredictor
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator_2,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)

class SamAutomaticMaskGenerator4Batch:
    def __init__(
            self,
            model: Sam,
            points_per_side: Optional[int] = 32,
            points_per_batch: int = 64,
            pred_iou_thresh: float = 0.88,
            stability_score_thresh: float = 0.95,
            stability_score_offset: float = 1.0,
            box_nms_thresh: float = 0.7,
            crop_n_layers: int = 0,
            crop_nms_thresh: float = 0.7,
            crop_overlap_ratio: float = 512 / 1500,
            crop_n_points_downscale_factor: int = 1,
            point_grids: Optional[List[np.ndarray]] = None,
            min_mask_region_area: int = 0,
            output_mode: str = "binary_mask",
    ):
        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictor(model)
        self.sam_model = model
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        self.orig_size = (720, 1280)
        self.transform_size = (576, 1024)
        self.crop_box = [0, 0, self.orig_size[1], self.orig_size[0]]
        self.layer_idx = 0

    # @torch.no_grad()
    def generate(self, image_batch, transformed_points_batch, points_labels):
        b, c, h, w = image_batch.shape
        with torch.no_grad():
            image_embedding = self.sam_model.image_encoder(image_batch)

        # points_scale = np.array([[self.orig_size[1], self.orig_size[0]]])
        # points_for_image = self.point_grids[self.layer_idx] * points_scale
        # points_for_image_batch = np.stack((points_for_image,)*b, axis=0)

        class_pred_batches_list = []
        for j in range(b):
            class_pred_one_batch_list = []
            image_embedding_batch = image_embedding[j].unsqueeze(0)
            transformed_points_pre_batch = transformed_points_batch[j]
            points_labels_pre_batch = points_labels[j]
            for i in range(0, len(transformed_points_pre_batch), self.points_per_batch):
                points_batch = transformed_points_pre_batch[i:i+self.points_per_batch, :]
                points_labels_batch = points_labels_pre_batch[i:i+self.points_per_batch]
                batch_data = self._process_batch(points_batch[:, None, :], points_labels_batch, image_embedding_batch)

                class_pred_one_batch_list.append(batch_data)

            class_preds_one_batch = torch.cat(class_pred_one_batch_list, dim=0)
            class_pred_batches_list.append(class_preds_one_batch)

        class_preds = torch.stack(class_pred_batches_list)

        return class_preds

    def _process_batch(self, point_coords, point_labels, image_embedding_features):
        points = (point_coords, point_labels)

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )

        low_res_masks, iou_preds, class_pred_feat = self.sam_model.mask_decoder(
            image_embeddings=image_embedding_features,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )

        return class_pred_feat