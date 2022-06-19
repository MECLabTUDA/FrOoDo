import torch
from torch.utils.data import Dataset
import numpy as np
import tqdm

from os.path import isfile, join, isdir
from os import listdir
from pathlib import Path
import json
import imageio
import copy

from ..utils import *


class MultiFileOverlappingTilesDataset(Dataset):
    def __init__(
        self,
        folder,
        tile_folder,
        size,
        dataset_name,
        mode,
        overlap,
        map_classes,
        ignore_classes,
        ood_classes,
        remain_classes,
        ignore_index,
        non_ignore_threshold,
        files=None,
        force_overwrite=False,
    ):
        self.folder = folder
        self.files = files
        self.dataset_name = dataset_name
        self.mode = mode
        self.size = size
        self.tile_folder = tile_folder
        self.overlap = overlap
        self.map_classes = map_classes
        self.ignore_classes = ignore_classes
        self.ood_classes = ood_classes
        self.remain_classes = remain_classes
        self.ignore_index = ignore_index
        self.non_ignore_threshold = non_ignore_threshold
        self.tiles = []
        self.valid = {}

        if not isdir(join(self.folder, "images")):
            raise Exception("No 'images' folder found")
        elif not isdir(join(self.folder, "masks")):
            raise Exception("No 'masks' folder found")

        images = listdir(join(self.folder, "images"))
        masks = listdir(join(self.folder, "images"))

        if len(images) == 0 or len(masks) == 0:
            raise Exception("The 'images' or 'masks' folder must not be empty")
        elif len(images) != len(masks):
            raise Exception(
                "The 'images' and 'masks' folder must have same number of elements"
            )

        if images[0][-4:] != ".tif" or masks[0][-4:] != ".tif":
            print("Use 'safe_as_tif' method to convert to tif")
            raise Exception("Images and Masks need to be in tif format")

        Path(join(self.tile_folder, self.dataset_name)).mkdir(
            parents=True, exist_ok=True
        )

        if self.files == None:
            self.files = masks

        # Load tile file
        self._set_tiles(self.files, {}, force_overwrite)

    def set_mode(self, mode):
        self.mode = mode

    def _set_tiles(self, files, config, force_overwrite=False):
        self.tiles, self.valid = self.get_tile_file(
            self._get_filename(), files, config, force_overwrite
        )
        return self

    def _get_filename(self):
        return f"{self.dataset_name}_{self.size[0]}_{self.size[1]}_{self.overlap}_{self.non_ignore_threshold}_mask.json"

    def _parse_data(self, data, files):
        tiles = []
        valid = []
        for f in files:
            for filetiles in data[f]["tiles"]:
                tiles.append([f, *filetiles])
            valid.append(data[f]["valid"])

        return tiles, np.sum(np.array(valid), axis=0)

    def get_tile_file(self, filename, files, config=None, force_overwrite=False):

        file = join(self.tile_folder, self.dataset_name, filename)

        if not isfile(file) or force_overwrite:
            # Create tile file if no file exists for specific config
            data = {}
            for image in tqdm.tqdm(files):
                til, val = get_valid_regions_of_image(
                    folder=join(self.folder, "masks"),
                    image=image,
                    ignore_classes=self.ignore_classes,
                    non_ignore_threshold=self.non_ignore_threshold,
                    overlap=self.overlap,
                    size=self.size,
                )
                data[image] = {"tiles": til, "valid": val}
            self.save_valid_regions(filename, data, config)
            return self._parse_data(data, files)

        with open(file, "r") as content:
            raw_tile_file = json.load(content)

        # print(f"Loaded tile file {file}")

        return self._parse_data(raw_tile_file["data"], files)

    def get_train_val_test_set(self, val_per=0.15, test_per=0.15, seed=42):

        random.Random(seed).shuffle(self.files)

        split_val = int(np.floor(val_per * len(self.files)))
        split_test = int(np.floor(test_per * len(self.files)))
        train_files, val_files, test_files = (
            self.files[split_val + split_test :],
            self.files[:split_val],
            self.files[split_val : split_val + split_test],
        )
        return (
            copy.deepcopy(self)._set_tiles(train_files, {}),
            copy.deepcopy(self)._set_tiles(val_files, {}),
            copy.deepcopy(self)._set_tiles(test_files, {}),
        )

    def save_valid_regions(self, filename, data, config):
        file = join(self.tile_folder, self.dataset_name, filename)
        print(
            f"Saving tile file at {join(self.tile_folder, self.dataset_name, filename)}"
        )
        data = {
            "config": config,
            "data": data,
        }
        with open(file, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
            )

    def __len__(self):
        return len(self.tiles)

    def _random_crop(self, img, mask, height, width):
        # print(img.shape)
        # print(img.shape, mask.shape, height, width)
        x = random.randint(0, img.shape[2] - width)
        y = random.randint(0, img.shape[1] - height)
        img = img[:, y : y + height, x : x + width]
        mask = mask[y : y + height, x : x + width]
        return (img, mask)

    def __getitem__(self, index):
        image = to_tensor(
            read_tif_region(
                join(self.folder, "images", self.tiles[index][0]),
                *self.tiles[index][1:],
            ).astype(np.float32)
            / 255.0
        )

        masks = apply_mask_changes(
            read_tif_region(
                join(self.folder, "masks", self.tiles[index][0]),
                *self.tiles[index][1:],
            ),
            map_classes=self.map_classes,
            ignore_classes=self.ignore_classes,
            ood_classes=self.ood_classes,
            ignore_index=self.ignore_index,
            remain_classes=self.remain_classes,
            mode=self.mode,
        )

        return image, masks

    def __str__(self):
        return f"Dataset '{self.dataset_name}' at location {self.folder} with {self.valid[0]} valid and {self.valid[1]} invalid tiles"

    @staticmethod
    def save_as_tif(raw_folder, tif_folder, open_func=imageio.imread):
        Path(join(tif_folder, "images")).mkdir(parents=True, exist_ok=True)
        Path(join(tif_folder, "masks")).mkdir(parents=True, exist_ok=True)
        for image in listdir(join(raw_folder, "images")):
            image_data = open_func(join(raw_folder, "images", image))
            mask_data = open_func(join(raw_folder, "masks", image))
            tiff.imwrite(
                join(tif_folder, "images", image[:-3] + "tif"),
                image_data,
                photometric="rgb",
            )
            tiff.imwrite(join(tif_folder, "masks", image[:-3] + "tif"), mask_data)
