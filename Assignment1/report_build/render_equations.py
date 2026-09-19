"""Render equations to transparent PNGs using matplotlib mathtext (no LaTeX install needed)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "equations")
os.makedirs(OUT_DIR, exist_ok=True)

EQUATIONS = {
    "net_input": r"$\mathrm{net} = \sum_{i=1}^{d} w_i x_i + b = \mathbf{w}^T\mathbf{x} + b$",
    "logistic": r"$f(\mathrm{net}) = \dfrac{1}{1+e^{-\mathrm{net}}}, \qquad f'(\mathrm{net}) = f(\mathrm{net})\,(1-f(\mathrm{net}))$",
    "tanh": r"$f(\mathrm{net}) = \tanh(\mathrm{net}), \qquad f'(\mathrm{net}) = 1 - f(\mathrm{net})^2$",
    "linear": r"$f(\mathrm{net}) = \mathrm{net}, \qquad f'(\mathrm{net}) = 1$",
    "error": r"$E = \dfrac{1}{2N}\sum_{k=1}^{N} (t_k - o_k)^2$",
    "update_rule": r"$w_i \leftarrow w_i + \eta\,(t-o)\,f'(\mathrm{net})\,x_i, \qquad b \leftarrow b + \eta\,(t-o)\,f'(\mathrm{net})$",
    "precision": r"$\mathrm{Precision}_j = \dfrac{CM_{jj}}{\sum_{i} CM_{ij}}$",
    "recall": r"$\mathrm{Recall}_i = \dfrac{CM_{ii}}{\sum_{j} CM_{ij}}$",
    "fmeasure": r"$F_i = \dfrac{2\,\mathrm{Precision}_i\,\mathrm{Recall}_i}{\mathrm{Precision}_i+\mathrm{Recall}_i}$",
    "rmse": r"$\mathrm{RMSE} = \sqrt{\dfrac{1}{N}\sum_{k=1}^{N}(t_k-o_k)^2}$",
    "pct_rmse": r"$\%\mathrm{RMSE} = \dfrac{\mathrm{RMSE}}{\overline{t}}\times 100$",
}


def render(name, tex, fontsize=16):
    from matplotlib import mathtext
    from matplotlib.font_manager import FontProperties
    out_path = os.path.join(OUT_DIR, f"{name}.png")
    mathtext.math_to_image(tex, out_path, dpi=250, prop=FontProperties(size=fontsize))
    return out_path


if __name__ == "__main__":
    for name, tex in EQUATIONS.items():
        p = render(name, tex)
        print(p)
