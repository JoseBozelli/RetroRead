"""
Verifies that COCO's "dial" category segmentation is the needle/pointer
mask (not a duplicate gauge-face mask), by decoding one and checking its
pixel count is small and localized -- consistent with a thin needle shape,
not a whole gauge face.

Run from the repo root with:
    uv run python scripts/exploratory/verify_dial_is_needle_mask.py
"""

import json

from pycocotools import mask as mask_utils

from retroread.config import ENDAVA_DS5_COCO


def main() -> None:
    with ENDAVA_DS5_COCO.open() as f:
        coco = json.load(f)

    dial_ann = next(a for a in coco["annotations"] if a.get("category_name") == "dial" and a["image_id"] == 1)

    print(f"dial bbox: {dial_ann['bbox']}")
    print(f"dial synth_dial_value: {dial_ann['synth_dial_value']}")

    seg = dict(dial_ann["segmentation"])
    if isinstance(seg["counts"], str):
        seg["counts"] = seg["counts"].encode("utf-8")

    mask = mask_utils.decode(seg)
    print(f"\nMask shape: {mask.shape}")
    print(f"Foreground pixel count: {mask.sum()} out of {mask.size} total ({100*mask.sum()/mask.size:.2f}%)")

    ys, xs = mask.nonzero()
    print(f"Foreground bounding extent: x=[{xs.min()},{xs.max()}] y=[{ys.min()},{ys.max()}]")
    print(f"That extent as width x height: {xs.max()-xs.min()} x {ys.max()-ys.min()}")


if __name__ == "__main__":
    main()