import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

from perceiver.data.audio.symbolic import SymbolicAudioDataModule
from perceiver.data.audio.utils import download_file, extract_file


class MaestroV3DataModule(SymbolicAudioDataModule):
    def __init__(
        self,
        *args: Any,
        dataset_uri: str = "https://martin-krasser.com/perceiver/data/midi/maestro-v3.0.0-midi.zip",
        dataset_dir: str = os.path.join(".cache", "maestro-v3-midi"),
        **kwargs: Any,
    ):
        super().__init__(dataset_dir, *args, **kwargs)
        self._dataset_uri = dataset_uri

    @property
    def source_dir(self) -> Path:
        return Path(self.hparams.dataset_dir) / "source"

    def load_source_dataset(self) -> Dict[str, Path]:
        if self.source_dir.exists():
            shutil.rmtree(self.source_dir)

        self.source_dir.mkdir(parents=True, exist_ok=False)

        download_dir = self.source_dir / "_download"
        download_dir.mkdir(parents=True, exist_ok=False)

        split_dir = self.source_dir / "_splits"
        train_dir = split_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=False)

        valid_dir = split_dir / "valid"
        valid_dir.mkdir(parents=True, exist_ok=False)

        dataset_file = download_dir / "maestro-v3.0.0-midi.zip"
        download_file(self._dataset_uri, dataset_file)
        extract_file(dataset_file, download_dir)

        dataset_dir = download_dir / "maestro-v3.0.0"
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Could not find Maestro v3 dataset directory in downloaded dataset (expected=`{dataset_dir}`)"
            )

        self._create_dataset_splits(dataset_dir, train_dir, valid_dir)
        shutil.rmtree(download_dir)

        return {"train": train_dir, "valid": valid_dir}

    @staticmethod
    def _create_dataset_splits(source_dataset_dir: Path, train_dir: Path, valid_dir: Path):
        meta_file = source_dataset_dir / "maestro-v3.0.0.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Could not find Maestro v3 dataset meta file (expected=`{meta_file}`)")

        with open(meta_file) as f:
            metadata = json.load(f)

        splits = {}
        for _id, file_path in metadata["midi_filename"].items():
            splits[file_path] = metadata["split"][_id]

        for file_path, split in splits.items():
            if split == "test":
                continue

            source_file = source_dataset_dir / file_path

            target_dir = train_dir if split == "train" else valid_dir
            target_file = target_dir / file_path

            if not target_file.parent.exists():
                target_file.parent.mkdir(parents=True, exist_ok=False)

            shutil.move(source_file, target_file)
