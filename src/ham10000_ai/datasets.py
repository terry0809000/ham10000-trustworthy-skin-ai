from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance


class HAM10000Sequence:
    """Minimal Keras-compatible CPU image loader.

    This avoids tf.data GPU-kernel issues and keeps image IO explicit. It returns
    ({"image": X}, y) or ({"image": X, "metadata": M}, y).
    """

    def __init__(self, df: pd.DataFrame, batch_size: int = 8, image_size: int = 224,
                 shuffle: bool = False, seed: int = 42, metadata: Optional[np.ndarray] = None):
        self.df = df.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.image_size = int(image_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self.metadata = metadata
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.stack([self._load_image(self.df.loc[i, "image_path"]) for i in batch_idx]).astype("float32")
        y = self.df.loc[batch_idx, "label"].to_numpy(dtype="int32")
        inputs = {"image": X}
        if self.metadata is not None:
            inputs["metadata"] = self.metadata[batch_idx].astype("float32")
        return inputs, y

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def _load_image(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        return np.asarray(img, dtype="float32")
