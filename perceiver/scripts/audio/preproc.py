import multiprocessing as mp

import jsonargparse

from perceiver.data.audio import GiantMidiPianoDataModule, MaestroV3DataModule

DATAMODULE_CLASSES = {"maestro-v3": MaestroV3DataModule, "giantmidi": GiantMidiPianoDataModule}


def main(args):
    dm_class = DATAMODULE_CLASSES[args.dataset]

    del args.dataset
    if args.dataset_uri is None:
        del args.dataset_uri
    if args.dataset_dir is None:
        del args.dataset_dir

    dm_class(**args, pin_memory=False).prepare_data()


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description="Preprocess dataset for training")
    parser.add_argument("dataset", default="maestro-v3", choices=list(DATAMODULE_CLASSES.keys()))
    parser.add_argument("--dataset_uri", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--max_seq_len", type=int, required=True)
    parser.add_argument("--preproc_workers", default=mp.cpu_count(), type=int)
    main(parser.parse_args())
