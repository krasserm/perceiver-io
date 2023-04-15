import os

import jsonargparse
from perceiver.model.audio import symbolic as sam

from perceiver.model.text import classifier as txt_clf, clm, mlm
from perceiver.model.vision import image_classifier as img_clf, optical_flow as opt_flow

GROUP_OFFICIAL_MODELS = "official-models"
GROUP_TRAINING_CHECKPOINTS = "training-checkpoints"
GROUP_ALL = "all"


def checkpoint_url(path, version="0.8.0"):
    return f"https://martin-krasser.com/perceiver/logs-{version}/{path}"


def convert_official_models(output_dir, **kwargs):
    mlm.convert_model(
        save_dir=os.path.join(output_dir, "perceiver-io-mlm"),
        source_repo_id="deepmind/language-perceiver",
        **kwargs,
    )
    img_clf.convert_model(
        save_dir=os.path.join(output_dir, "perceiver-io-img-clf"),
        source_repo_id="deepmind/vision-perceiver-fourier",
        **kwargs,
    )
    opt_flow.convert_model(
        save_dir=os.path.join(output_dir, "perceiver-io-optical-flow"),
        source_repo_id="deepmind/optical-flow-perceiver",
        **kwargs,
    )


def convert_training_checkpoints(output_dir, **kwargs):
    clm.convert_checkpoint(
        save_dir=os.path.join(output_dir, "perceiver-ar-clm-base"),
        ckpt_url=checkpoint_url("clm-fsdp/version_1/checkpoints/epoch=000-val_loss=2.820.ckpt"),
        tokenizer_name="xlnet-base-cased",
        **kwargs,
    )
    mlm.convert_checkpoint(
        save_dir=os.path.join(output_dir, "perceiver-io-mlm-imdb"),
        ckpt_url=checkpoint_url("mlm/version_0/checkpoints/epoch=012-val_loss=1.165.ckpt"),
        tokenizer_name="krasserm/perceiver-io-mlm",
        **kwargs,
    )
    txt_clf.convert_imdb_classifier_checkpoint(
        save_dir=os.path.join(output_dir, "perceiver-io-txt-clf-imdb"),
        ckpt_url=checkpoint_url("txt_clf/version_1/checkpoints/epoch=006-val_loss=0.156.ckpt"),
        tokenizer_name="krasserm/perceiver-io-mlm",
        **kwargs,
    )
    img_clf.convert_mnist_classifier_checkpoint(
        save_dir=os.path.join(output_dir, "perceiver-io-img-clf-mnist"),
        ckpt_url=checkpoint_url("img_clf/version_0/checkpoints/epoch=025-val_loss=0.065.ckpt"),
        **kwargs,
    )

    sam.convert_checkpoint(
        save_dir=os.path.join(output_dir, "perceiver-ar-sam-giant-midi"),
        ckpt_url=checkpoint_url("sam/version_1/checkpoints/epoch=027-val_loss=1.944.ckpt"),
        **kwargs,
    )


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description="Convert official models and training checkpoint")
    parser.add_argument("group", default="all", choices=[GROUP_ALL, GROUP_OFFICIAL_MODELS, GROUP_TRAINING_CHECKPOINTS])
    parser.add_argument("--output_dir", default="krasserm")
    parser.add_argument("--push_to_hub", default=False, type=bool)
    parser.add_argument("--commit_message", default=None)

    args = parser.parse_args()

    if args.group in [GROUP_ALL, GROUP_OFFICIAL_MODELS]:
        convert_official_models(
            output_dir=args.output_dir,
            push_to_hub=args.push_to_hub,
            commit_message=args.commit_message,
        )

    if args.group in [GROUP_ALL, GROUP_TRAINING_CHECKPOINTS]:
        convert_training_checkpoints(
            output_dir=args.output_dir,
            push_to_hub=args.push_to_hub,
            commit_message=args.commit_message,
        )
