__version__ = "0.1.0"
__author__ = "Martin Krasser"
__author_email__ = "krasserm@googlemail.com"
__license__ = "Apache-2.0"
__homepage__ = ("https://github.com/krasserm/perceiver-io",)
__copyright__ = "Copyright (c) 2021-2022, %s." % __author__
__doc__ = 'Perceiver IO'
__long_doc__ = (
    "# %s" % __doc__
    + """

A PyTorch implementation of

- [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)
- [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795)

This project supports training of Perceiver IO models with [Pytorch Lightning](https://www.pytorchlightning.ai/).
Training examples are given in section [Tasks](#tasks), inference examples in section [Notebooks](#notebooks).
Perceiver IO models are constructed with generic encoder and decoder classes and task-specific input and
output adapters (see [Model API](#model-api)). The command line interface is implemented with
[Lighting CLI](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html).
"""
)
