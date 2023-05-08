import os
import shutil
from pathlib import Path
from typing import Any, Dict

from perceiver.data.audio.symbolic import SymbolicAudioDataModule
from perceiver.data.audio.utils import download_file, extract_file


class GiantMidiPianoDataModule(SymbolicAudioDataModule):
    def __init__(
        self,
        *args: Any,
        dataset_uri: str = "https://martin-krasser.com/perceiver/data/midi/giantmidi-piano.zip",
        dataset_dir: str = os.path.join(".cache", "giantmidi-piano"),
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

        dataset_file = download_dir / "giantmidi-piano.zip"
        download_file(self._dataset_uri, dataset_file)
        extract_file(dataset_file, download_dir)

        train_dir = download_dir / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"Could not find training directory in downloaded dataset (expected=`{train_dir}`)")

        valid_dir = download_dir / "valid"
        if not valid_dir.exists():
            raise FileNotFoundError(
                f"Could not find validation directory in downloaded dataset (expected=`{valid_dir}`)"
            )

        return {"train": train_dir, "valid": valid_dir}
