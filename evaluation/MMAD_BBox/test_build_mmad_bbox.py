import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import build_mmad_bbox as builder


class GroundTruthTests(unittest.TestCase):
    def test_diagonal_connectivity_and_single_pixel_at_image_boundary(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[0, 0] = mask[1, 1] = mask[6, 6] = True
        self.assertEqual(builder.boxes_from_mask(mask), [[0, 0, 2, 2], [6, 6, 7, 7]])

    def test_rgb_foreground_and_union_of_loco_instance_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            red = np.zeros((4, 5, 3), dtype=np.uint8)
            red[0, 1] = [255, 0, 0]
            Image.fromarray(red).save(root / "000.png")
            labels = np.zeros((4, 5), dtype=np.uint8)
            labels[3, 4] = 242
            Image.fromarray(labels).save(root / "001.png")
            mask = builder.load_mask(sorted(root.glob("*.png")), (5, 4))
            self.assertEqual(int(mask.sum()), 2)
            self.assertEqual(builder.boxes_from_mask(mask), [[1, 0, 2, 1], [4, 3, 5, 4]])

    def test_loco_png_annotation_resolves_to_existing_mask_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "MVTec-LOCO/breakfast_box/ground_truth/structural_anomalies/001"
            directory.mkdir(parents=True)
            Image.new("L", (3, 2), 255).save(directory / "000.png")
            resolved, files = builder.resolve_mask(
                root, "MVTec-LOCO/breakfast_box/test/structural_anomalies/001.png",
                "ground_truth/structural_anomalies/001.png",
            )
            self.assertEqual(resolved, directory)
            self.assertEqual(files, [directory / "000.png"])

    def test_mismatched_mask_size_fails_instead_of_changing_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.png"
            Image.new("L", (2, 2), 255).save(path)
            with self.assertRaisesRegex(ValueError, "size"):
                builder.load_mask([path], (3, 2))


if __name__ == "__main__":
    unittest.main()
