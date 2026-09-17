from pathlib import Path
import io
from zipfile import ZipFile
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from .preprocessing import preprocess_array,MODES


def prepare_archive(archive, destination):
    """Extract only the known dataset structure, never arbitrary archive paths."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as outer:
        for name in ('crop_metadata_index.csv', 'source_image_metadata.csv'):
            (destination / name).write_bytes(outer.read('DATASETS/' + name))
        payload = outer.read('DATASETS/images.zip')
    expected = set(pd.read_csv(destination / 'crop_metadata_index.csv').filename)
    seen = set()
    image_dir = destination / 'images'
    image_dir.mkdir(exist_ok=True)
    with ZipFile(io.BytesIO(payload)) as inner:
        for entry in inner.infolist():
            if entry.is_dir():
                continue
            name = Path(entry.filename).name
            if name not in expected or name in seen:
                raise ValueError(f'Unexpected or duplicate image: {name}')
            seen.add(name)
            target = image_dir / name
            if not target.exists():
                target.write_bytes(inner.read(entry))
    if seen != expected:
        raise ValueError('Archive and crop index do not match')
    return destination


def load_manifest(data_dir, seed=2026):
    data_dir = Path(data_dir)
    crops = pd.read_csv(data_dir / 'crop_metadata_index.csv')
    sources = pd.read_csv(data_dir / 'source_image_metadata.csv')
    if not crops.filename.is_unique or not sources.source_image_id.is_unique:
        raise ValueError('Duplicate metadata join keys')
    frame = crops.merge(sources, on='source_image_id', how='left', validate='many_to_one')
    if frame.isna().any().any():
        raise ValueError('Missing metadata')
    groups = np.array(sorted(frame.source_image_id.unique()))
    if len(groups) < 10:
        raise ValueError('At least 10 source groups required')
    np.random.default_rng(seed).shuffle(groups)
    n_train, n_val = int(.7 * len(groups)), int(.15 * len(groups))
    assignment = {g: ('train' if i < n_train else 'validation' if i < n_train+n_val else 'calibration') for i,g in enumerate(groups)}
    frame['split'] = frame.source_image_id.map(assignment)
    frame['path'] = [str(data_dir / 'images' / name) for name in frame.filename]
    if not all(Path(p).is_file() for p in frame.path):
        raise FileNotFoundError('Some indexed image files are missing')
    return frame


class CropDataset(Dataset):
    def __init__(self, frame, preprocessing='raw'):
        self.frame = frame.reset_index(drop=True)
        if preprocessing not in MODES:raise ValueError('Unknown preprocessing mode')
        self.preprocessing=preprocessing

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            if image.mode != 'L' or image.size != (227,227):
                raise ValueError(f'Unexpected image contract: {row.filename}')
            array = np.array(image, dtype=np.float32) / 255.
        return torch.from_numpy(preprocess_array(array,self.preprocessing)).unsqueeze(0), index
