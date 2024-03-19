import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple

from .modeling import Sam
from .predictor import SamPredictor
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
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
        self.point_grids = build_all_layer_point_grids(32, 0, 1)
        self.device = torch.device("cuda:0")

    def compute_iou(self, mask1, mask2):
        intersect = (mask1 & mask2).sum()
        union = mask1.sum() + mask2.sum() - intersect
        iou = intersect / union
        return iou

    @torch.no_grad()
    def generate(self, image_batch, labels_batch):
        b, c, h, w = image_batch.shape
        image_embedding = self.sam_model.image_encoder(image_batch)

        # points_scale = np.array([[self.orig_size[1], self.orig_size[0]]])
        # points_for_image = self.point_grids[self.layer_idx] * points_scale
        # points_for_image_batch = np.stack((points_for_image,)*b, axis=0)

        # data = MaskData()
        # for i in range(0, len(transformed_points_batch[1]), self.points_per_batch):
        #     points_batch = transformed_points_batch[:, i:i+self.points_per_batch, :]
        #     points_labels_batch = points_labels[:, i:i+self.points_per_batch]
        #     batch_data = self._process_batch(points_batch[:, :, None, :], points_labels_batch[:, :, None])

        smoke_ious_list = []
        iou_token_outs_list = []
        for i in range(b):
            image_embedding_feat = image_embedding[i].unsqueeze(0)
            labels = labels_batch[0].numpy()

            points_scale = np.array(self.orig_size)[None, ::-1]
            points_for_image = self.point_grids[0] * points_scale

            data = MaskData()
            for (points,) in batch_iterator(self.points_per_batch, points_for_image):
                batch_data = self._process_batch(points, image_embedding_feat)
                data.cat(batch_data)
                del batch_data

            # Remove duplicates within this crop.
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                data["iou_preds"],
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.box_nms_thresh,
            )
            data.filter(keep_by_nms)

            # Return to the original image frame
            data["boxes"] = uncrop_boxes_xyxy(data["boxes"], [0, 0, self.orig_size[1], self.orig_size[0]])
            data["points"] = uncrop_points(data["points"], [0, 0, self.orig_size[1], self.orig_size[0]])
            data["crop_boxes"] = torch.tensor([[0, 0, self.orig_size[1], self.orig_size[0]] for _ in range(len(data["rles"]))])

            data.to_numpy()

            data["segmentations"] = [rle_to_mask(rle) for rle in data["rles"]]

            curr_anns = []
            for idx in range(len(data["segmentations"])):
                ann = {
                    "segmentation": data["segmentations"][idx],
                    "area": area_from_rle(data["rles"][idx]),
                    "bbox": box_xyxy_to_xywh(data["boxes"][idx]).tolist(),
                    "predicted_iou": data["iou_preds"][idx].item(),
                    "point_coords": [data["points"][idx].tolist()],
                    "stability_score": data["stability_score"][idx].item(),
                    "crop_box": box_xyxy_to_xywh(data["crop_boxes"][idx]).tolist(),
                    "iou_token_out": data["iou_token_out"][idx],
                }
                curr_anns.append(ann)

            smoke_ious = []
            iou_token_outs = []
            for mask in curr_anns:
                smoke_iou = self.compute_iou(mask['segmentation'], labels)
                smoke_ious.append(torch.Tensor([smoke_iou]))
                iou_token_outs.append(torch.from_numpy(mask['iou_token_out']))
            smoke_ious = torch.stack(smoke_ious)
            iou_token_outs = torch.stack(iou_token_outs)

            smoke_ious_list.append(smoke_ious)
            iou_token_outs_list.append(iou_token_outs)

        return smoke_ious_list, iou_token_outs_list, curr_anns

    def _process_batch(self, points, image_embedding_features):
        transformed_points = self.predictor.transform.apply_coords(points, self.orig_size)
        in_points = torch.as_tensor(transformed_points).to(self.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int).to(self.device)

        prompt_points = (in_points[:, None, :], in_labels[:, None])
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=prompt_points,
            boxes=None,
            masks=None,
        )

        low_res_masks, iou_preds, iou_token_out = self.sam_model.mask_decoder(
            image_embeddings=image_embedding_features,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )

        masks = self.sam_model.postprocess_masks(low_res_masks, self.transform_size, self.orig_size)

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            iou_token_out=iou_token_out.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], [0, 0, self.orig_size[1], self.orig_size[0]], [0, 0, self.orig_size[1], self.orig_size[0]])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], [0, 0, self.orig_size[1], self.orig_size[0]], self.orig_size[0], self.orig_size[1])
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data