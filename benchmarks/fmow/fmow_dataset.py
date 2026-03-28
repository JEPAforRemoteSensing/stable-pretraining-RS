"""fMoW dual-modality streaming dataset for stable-pretraining.

Wraps Lightning StreamingDataset to return dict-format batches with
separate RGB and multispectral (MS) image tensors.
"""

import numpy as np
import torch
from lightning.data import StreamingDataset

import stable_pretraining as spt

CAT_TO_IDX = {
    "airport": 0,
    "airport_hangar": 1,
    "airport_terminal": 2,
    "amusement_park": 3,
    "aquaculture": 4,
    "archaeological_site": 5,
    "barn": 6,
    "border_checkpoint": 7,
    "burial_site": 8,
    "car_dealership": 9,
    "construction_site": 10,
    "crop_field": 11,
    "dam": 12,
    "debris_or_rubble": 13,
    "educational_institution": 14,
    "electric_substation": 15,
    "factory_or_powerplant": 16,
    "fire_station": 17,
    "flooded_road": 18,
    "fountain": 19,
    "gas_station": 20,
    "golf_course": 21,
    "ground_transportation_station": 22,
    "helipad": 23,
    "hospital": 24,
    "impoverished_settlement": 25,
    "interchange": 26,
    "lake_or_pond": 27,
    "lighthouse": 28,
    "military_facility": 29,
    "multi-unit_residential": 30,
    "nuclear_powerplant": 31,
    "office_building": 32,
    "oil_or_gas_facility": 33,
    "park": 34,
    "parking_lot_or_garage": 35,
    "place_of_worship": 36,
    "police_station": 37,
    "port": 38,
    "prison": 39,
    "race_track": 40,
    "railway_bridge": 41,
    "recreational_facility": 42,
    "road_bridge": 43,
    "runway": 44,
    "shipyard": 45,
    "shopping_mall": 46,
    "single-unit_residential": 47,
    "smokestack": 48,
    "solar_farm": 49,
    "space_facility": 50,
    "stadium": 51,
    "storage_tank": 52,
    "surface_mine": 53,
    "swimming_pool": 54,
    "toll_booth": 55,
    "tower": 56,
    "tunnel_opening": 57,
    "waste_disposal": 58,
    "water_treatment_facility": 59,
    "wind_farm": 60,
    "zoo": 61,
}

NUM_CLASSES = len(CAT_TO_IDX)


class FMoWStreamingDataset(StreamingDataset):
    """fMoW dual-modality dataset backed by Lightning Streaming.

    During training (num_views > 1), returns stacked multi-view tensors.
    During eval (num_views == 1), returns single-view tensors.

    Args:
        input_dir: Path to Lightning streaming data directory.
        transform_rgb: Albumentations transform for RGB images.
        transform_ms: Albumentations transform for multispectral images.
        num_views: Number of augmented views per modality. Use 1 for eval.
        **kwargs: Forwarded to ``StreamingDataset`` (e.g. ``shuffle``, ``subsample``).

    Returns:
        dict with keys:
            - ``image_rgb``: ``(V, 3, H, W)`` if training, ``(3, H, W)`` if eval
            - ``image_ms``:  ``(V, 10, H, W)`` if training, ``(10, H, W)`` if eval
            - ``label``:     integer class label
            - ``sample_idx``: dataset index (needed by retrieval callbacks)
    """

    def __init__(
        self,
        input_dir,
        transform_rgb,
        transform_ms,
        num_views=4,
        **kwargs,
    ):
        super().__init__(input_dir=input_dir, **kwargs)
        self.transform_rgb = transform_rgb
        self.transform_ms = transform_ms
        self.num_views = num_views

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        rgb_np = np.array(data["rgb"], dtype=np.uint8)  # (H, W, 3)
        ms_np = np.array(data["ms"], dtype=np.uint8)  # (H, W, 10)
        label = CAT_TO_IDX[data["category"]]

        if self.num_views > 1:
            views_rgb = torch.stack(
                [self.transform_rgb(image=rgb_np)["image"] for _ in range(self.num_views)]
            )
            views_ms = torch.stack(
                [self.transform_ms(image=ms_np)["image"] for _ in range(self.num_views)]
            )
        else:
            views_rgb = self.transform_rgb(image=rgb_np)["image"]
            views_ms = self.transform_ms(image=ms_np)["image"]

        return {
            "image_rgb": views_rgb,
            "image_ms": views_ms,
            "label": label,
            "sample_idx": idx,
        }
