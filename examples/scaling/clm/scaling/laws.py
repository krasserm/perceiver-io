import functools
from dataclasses import dataclass

from scipy.optimize import curve_fit


@dataclass
class ScalingLaw:
    a: float
    b: float
    k_n: float
    k_d: float

    def n_opt(self, flops):
        return self.k_n * flops**self.a

    def d_opt(self, flops):
        return self.k_d * flops**self.b

    def __str__(self):
        return f"N_opt = {self.k_n:.4f} * C ** {self.a:.2f}\n" f"D_opt = {self.k_d:.4f} * C ** {self.b:.2f}"


def fit_scaling_law(flops_arr, params_arr, tokens_arr, a, b) -> ScalingLaw:
    k_n = fit_power_law(flops_arr, params_arr, m=a)
    k_d = fit_power_law(flops_arr, tokens_arr, m=b)
    return ScalingLaw(a=a, b=b, k_n=k_n, k_d=k_d)


def fit_power_law(xs, ys, m, k0=0.5):
    return curve_fit(functools.partial(power_law, m=m), xs, ys, p0=[k0])[0][0]


def power_law(x, k, m):
    return k * x**m
