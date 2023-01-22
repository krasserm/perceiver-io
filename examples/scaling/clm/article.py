from typing import Sequence

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from scaling.flops import ModelInfo, num_training_steps


def experiment_setup(models: Sequence[ModelInfo], c_ref: int, batch_size: int):
    """Return information related to training `models` under compute budget `c_ref`."""

    data = []

    for m in models:
        c_self_per_token = m.self_attn_flops()

        d_iso = c_ref / c_self_per_token
        s_iso = num_training_steps(
            num_tokens=d_iso,
            num_latents=m.num_latents,
            batch_size=batch_size,
        )

        c_self_approx = m.self_attn_flops_approx() * d_iso
        c_self = c_self_per_token * d_iso
        c_cross = m.cross_attn_flops() * d_iso
        c = c_self + c_cross

        n_self = m.num_self_attn_params()
        n_cross = m.num_cross_attn_params()
        n = n_self + n_cross

        data.append([m.num_channels, m.num_layers, s_iso, d_iso, n, n_cross, n_self, c, c_cross, c_self, c_self_approx])

    df = pd.DataFrame(
        data,
        columns=[
            "num_channels",
            "num_layers",
            "num_steps",
            "$D_{iso}$",
            "$N$",
            "$N_{cross}$",
            "$N_{self}$",
            "$C$",
            "$C_{cross}$",
            "$C_{self}$",
            r"$\hat{C}_{self}$",
        ],
    )
    df.index += 1

    format_spec = ["{:}", "{:}", "{:}", "{:.2e}", "{:.2e}", "{:.2e}", "{:.2e}", "{:.2e}", "{:.2e}", "{:.2e}", "{:.2e}"]
    return df.style.format(dict(zip(df.columns, format_spec)))


def experiment_ratios(models: Sequence[ModelInfo]):
    """Return compute- and parameter-related ratios, independent of compute budget."""

    data = []

    for m in models:
        c_self_approx_per_token = m.self_attn_flops_approx()
        c_self_per_token = m.self_attn_flops()
        c_cross_per_token = m.cross_attn_flops()

        c_self_approx_ratio = c_self_per_token / c_self_approx_per_token
        c_cross_contrib = c_cross_per_token / (c_cross_per_token + c_self_per_token)

        n_self = m.num_self_attn_params()
        n_cross = m.num_cross_attn_params()
        n_cross_contrib = n_cross / (n_cross + n_self)

        data.append([n_cross_contrib, c_cross_contrib, c_self_approx_ratio])

    df = pd.DataFrame(
        data, columns=[r"$N_{cross} \over N$", r"$C_{cross} \over C$", r"${C_{self}} \over {\hat{C}_{self}}$"]
    )
    df.index += 1

    format_spec = ["{:.4f}", "{:.4f}", "{:.4f}"]
    return df.style.format(dict(zip(df.columns, format_spec)))


def plot_experiment(models, model_labels, experiment_name):
    for i, (m, label) in enumerate(zip(models, model_labels)):
        df = pd.read_csv(f"data/validation/run-logs-0.8.0_{experiment_name}_version_{i}-tag-val_loss.csv")
        c_cross = m.cross_attn_flops()
        c_self = m.self_attn_flops()
        n_self = m.num_self_attn_params()
        y = df["Value"].to_numpy()
        x = np.linspace(0, (c_cross + c_self) / c_self, len(y) + 1)
        plt.plot(x[1:], y, label=f"{label}: $N_{{self}}$ = {n_self / 1e6:.1f}M, final loss = {y[-1]:.3f}")

    plt.axvline(1.0, ls="--", lw=0.5, c="black", label="$C_{self}$")
    plt.xlabel("FLOPs * $C_{self}$")
    plt.ylabel("Loss")
    plt.legend()
