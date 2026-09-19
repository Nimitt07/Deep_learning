"""Builds the plots, tables and confusion matrices from the saved training runs.

    python analyze.py            # reads results/, writes into ../report/{figures,tables}

Figures (one per architecture):  loss_<arch>.pdf
    top    : training error (a) and validation error (b), all seven optimizers superimposed
    bottom : training (solid) and validation (dashed) error of each optimizer separately
Figure of the best model:        confusion_best.pdf
Tables:  epochs.tex, accuracy.tex, final_error.tex, architectures.tex
"""
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter, LogLocator

import fcnn
from data_loader import load_data
from train import ARCHITECTURES, RESULTS, layer_sizes

HERE = os.path.dirname(os.path.abspath(__file__))
OPT_ORDER = ["SGD", "BatchGD", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam"]
OPT_LABEL = {"SGD": "SGD", "BatchGD": "Batch GD", "Momentum": "Momentum (GDR)",
             "NAG": "NAG", "AdaGrad": "AdaGrad", "RMSProp": "RMSProp", "Adam": "Adam"}
OPT_SHORT = {"SGD": "SGD", "BatchGD": "BGD", "Momentum": "GDR", "NAG": "NAG",
             "AdaGrad": "AdaGrad", "RMSProp": "RMSProp", "Adam": "Adam"}
OPT_BATCH = {"SGD": "1", "BatchGD": "N", "Momentum": "1", "NAG": "1",
             "AdaGrad": "N", "RMSProp": "N", "Adam": "1"}
# fixed categorical order (validated palette) + line style as secondary encoding
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
STYLES = ["-", "--", "-", (0, (5, 1.5)), "-.", ":", "-"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 7.5, "legend.fontsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.edgecolor": "#898781", "axes.linewidth": 0.6, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300,
})


def arch_name(arch):
    return "784-" + "-".join(map(str, ARCHITECTURES[arch])) + "-5"


def load_runs(res_dir):
    runs = {}
    for arch in ARCHITECTURES:
        for opt in OPT_ORDER:
            path = os.path.join(res_dir, f"{arch}_{opt}")
            if os.path.exists(path + ".json") and os.path.exists(path + ".npz"):
                with open(path + ".json") as f:
                    s = json.load(f)
                h = np.load(path + ".npz")
                s["hist"] = {k: h[k] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
                s["theta"] = h["theta"]
                runs[(arch, opt)] = s
    return runs


def _style(ax):
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _ylim(runs, arch):
    keys = [(arch, o) for o in OPT_ORDER if (arch, o) in runs]
    lo = min(min(runs[k]["hist"]["train_loss"].min(), runs[k]["hist"]["val_loss"].min()) for k in keys)
    hi = max(max(runs[k]["hist"]["train_loss"].max(), runs[k]["hist"]["val_loss"].max()) for k in keys)
    return 10 ** np.floor(np.log10(max(lo, 1e-9))), hi * 1.8


def _symlog_epochs(ax, max_ep):
    ax.set_xscale("symlog", linthresh=1, linscale=0.35)
    ax.set_xlim(-0.05, max_ep * 1.15)
    ticks = [0, 1, 5, 10, 50, 100, 500, 2000, 10000]
    ax.xaxis.set_major_locator(FixedLocator([t for t in ticks if t <= max_ep * 1.15]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def plot_architecture(runs, arch, max_ep, fig_dir):
    """(a) training error and (b) validation error of all optimizers superimposed,
    (c)-(i) training (solid) and validation (dashed) error of each optimizer separately."""
    opts = [o for o in OPT_ORDER if (arch, o) in runs]
    ylo, yhi = _ylim(runs, arch)

    fig = plt.figure(figsize=(4.8, 4.75))
    gs = fig.add_gridspec(3, 4, height_ratios=[2.0, 1, 1], hspace=0.55, wspace=0.3,
                          left=0.085, right=0.99, top=0.885, bottom=0.075)

    # ---- (a) training and (b) validation error, all optimizers ---------------------
    for p, (key, label) in enumerate((("train_loss", "training"), ("val_loss", "validation"))):
        ax = fig.add_subplot(gs[0, 0:2] if p == 0 else gs[0, 2:4])
        e0 = runs[(arch, opts[0])]["hist"][key][0]
        for opt in opts:
            i = OPT_ORDER.index(opt)
            e = runs[(arch, opt)]["hist"][key]
            noisy = opt == "Adam"
            ax.plot(np.arange(len(e)), e, color=COLORS[i], ls=STYLES[i],
                    lw=0.8 if noisy else 1.15, zorder=2 if noisy else 3)
            ax.plot(len(e) - 1, e[-1], marker=MARKERS[i], ms=3.8, color=COLORS[i],
                    mec="white", mew=0.5, zorder=4)
        ax.plot(0, e0, "o", ms=4.2, color=INK, mec="white", mew=0.7, zorder=5)
        ax.annotate(f"$E(0)$={e0:.4f}", (0, e0), xytext=(6, 2), textcoords="offset points",
                    fontsize=6.3, color=INK)
        _style(ax)
        ax.set_ylim(ylo, yhi * 3)
        _symlog_epochs(ax, max_ep)
        ax.set_xlabel("epoch (symlog scale)", labelpad=1)
        if p == 0:
            ax.set_ylabel("average error $E$", labelpad=1)
        ax.set_title(f"({'ab'[p]}) {label} error, all optimizers", loc="left",
                     fontweight="bold", fontsize=7.4, pad=2)

    handles = [matplotlib.lines.Line2D([], [], color=COLORS[i], ls=STYLES[i], lw=1.2,
                                       marker=MARKERS[i], ms=3.8, mec="white", mew=0.5,
                                       label=OPT_LABEL[o]) for i, o in enumerate(OPT_ORDER)]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, handlelength=2.6,
               columnspacing=1.0, labelspacing=0.2, fontsize=7, bbox_to_anchor=(0.54, 1.0))

    # ---- (c)-(i) one optimizer per panel: training solid, validation dashed ----------
    for k, opt in enumerate(opts):
        i = OPT_ORDER.index(opt)
        r = runs[(arch, opt)]
        tr, va = r["hist"]["train_loss"], r["hist"]["val_loss"]
        ep = np.arange(len(tr))
        ax = fig.add_subplot(gs[1 + k // 4, k % 4])
        ax.plot(ep, va, color=COLORS[i], ls=(0, (2.2, 1.2)), lw=0.9, alpha=0.75)
        ax.plot(ep, tr, color=COLORS[i], ls="-", lw=1.0)
        ax.plot(ep[-1], tr[-1], marker=MARKERS[i], ms=3.2, color=COLORS[i], mec="white", mew=0.4)
        _style(ax)
        ax.set_ylim(ylo, yhi)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.set_xlim(0, max(ep[-1], 1) * 1.04)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, integer=True))
        ax.tick_params(pad=1.2, length=2, labelsize=5.8)
        ax.set_title(f"({chr(99 + k)}) {OPT_LABEL[opt].replace('Momentum (GDR)', 'GDR')}: "
                     f"{r['epochs']} ep.", loc="left", fontsize=6.6, pad=1.5)
        if k % 4 == 0:
            ax.set_ylabel("$E$", labelpad=0)
        if k >= len(opts) - 4:
            ax.set_xlabel("epoch", labelpad=0.5, fontsize=6.5)

    # ---- key ---------------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    ax.plot([0.02, 0.22], [0.9, 0.9], color=MUTED, lw=1.0, transform=ax.transAxes)
    ax.plot([0.02, 0.22], [0.72, 0.72], color=MUTED, lw=0.9, ls=(0, (2.2, 1.2)),
            transform=ax.transAxes)
    ax.text(0.27, 0.9, "training", va="center", fontsize=6.3, transform=ax.transAxes)
    ax.text(0.27, 0.72, "validation", va="center", fontsize=6.3, transform=ax.transAxes)
    ax.text(0.02, 0.5, f"{arch}: {arch_name(arch)}\n"
            f"end marker = stopping epoch\n(c)-(i): linear epoch axis",
            va="top", fontsize=5.9, color=INK, linespacing=1.35, transform=ax.transAxes)

    fig.savefig(os.path.join(fig_dir, f"loss_{arch}.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def confusion(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def plot_confusions(cms, digits, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.25))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f7fafe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    for ax, (title, cm) in zip(axes, cms):
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else INK)
        ax.set_xticks(range(len(digits)), [str(d) for d in digits])
        ax.set_yticks(range(len(digits)), [str(d) for d in digits])
        ax.set_xlabel("predicted digit")
        ax.set_ylabel("true digit")
        ax.set_title(title, fontweight="bold")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(os.path.join(fig_dir, "confusion_best.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def pct(x):
    return f"{100 * x:.2f}"


def fmt_loss(x):
    """4 decimals, or scientific notation for errors that would round to 0.0000."""
    if x >= 5e-5:
        return f"{x:.4f}"
    mant, exp = f"{x:.1e}".split("e")
    return rf"${mant}{{\times}}10^{{{int(exp)}}}$"


def thin(n):
    return f"{n:,}".replace(",", r"\,")


def write_tables(runs, best, tab_dir, n_train):
    archs = [a for a in ARCHITECTURES if any((a, o) in runs for o in OPT_ORDER)]
    ncol = len(OPT_ORDER)
    opts_hdr = " & ".join(OPT_SHORT[o] for o in OPT_ORDER)
    bs_hdr = " & ".join(rf"{{\scriptsize $B{{=}}{OPT_BATCH[o]}$}}" for o in OPT_ORDER)

    def get(a, o):
        return runs.get((a, o))

    # ---- architectures -------------------------------------------------------------
    lines = [r"\begin{tabular}{llcr}", r"\toprule",
             r"Name & Layer sizes & Hidden layers & Parameters \\", r"\midrule"]
    for a in ARCHITECTURES:
        n_p = fcnn.Layout(layer_sizes(a)).n_params
        lines.append(f"{a} & {arch_name(a).replace('-', '--')} & {len(ARCHITECTURES[a])} & {thin(n_p)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(tab_dir, "architectures.tex"), "w").write("\n".join(lines) + "\n")

    # ---- epochs to convergence (rows: architectures) ---------------------------------
    lines = [r"\begin{tabular}{l" + "r" * ncol + "}", r"\toprule",
             f"Arch. & {opts_hdr} \\\\", f" & {bs_hdr} \\\\", r"\midrule"]
    for a in archs:
        vals = [get(a, o)["epochs"] if get(a, o) else None for o in OPT_ORDER]
        best_ep = min(v for v in vals if v is not None)
        cells = []
        for o, v in zip(OPT_ORDER, vals):
            if v is None:
                cells.append("--")
                continue
            c = f"{v}" + ("" if get(a, o)["converged"] else r"$^\dagger$")
            cells.append(r"\textbf{" + c + "}" if v == best_ep else c)
        lines.append(f"{a} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(tab_dir, "epochs.tex"), "w").write("\n".join(lines) + "\n")

    # ---- final error and number of weight updates ------------------------------------
    lines = [r"\begin{tabular}{l" + "r" * ncol + "}", r"\toprule",
             f"Arch. & {opts_hdr} \\\\", f" & {bs_hdr} \\\\", r"\midrule",
             rf"\multicolumn{{{ncol + 1}}}{{l}}{{\emph{{Average training error $E$ at the stopping epoch}}}} \\"]
    for a in archs:
        cells = [fmt_loss(get(a, o)["final_loss"]) if get(a, o) else "--" for o in OPT_ORDER]
        lines.append(f"{a} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule",
              rf"\multicolumn{{{ncol + 1}}}{{l}}{{\emph{{Number of weight updates (iterations)}}}} \\"]
    for a in archs:
        cells = []
        for o in OPT_ORDER:
            r = get(a, o)
            cells.append(thin(r["epochs"] * (n_train // r["batch_size"])) if r else "--")
        lines.append(f"{a} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(tab_dir, "final_error.tex"), "w").write("\n".join(lines) + "\n")

    # ---- training / validation accuracy ----------------------------------------------
    lines = [r"\begin{tabular}{ll" + "r" * ncol + "}", r"\toprule",
             f"Arch. & Set & {opts_hdr} \\\\", f" & & {bs_hdr} \\\\", r"\midrule"]
    for j, a in enumerate(archs):
        best_val = max(get(a, o)["val_acc"] for o in OPT_ORDER if get(a, o))
        tr = [pct(get(a, o)["train_acc"]) if get(a, o) else "--" for o in OPT_ORDER]
        va = []
        for o in OPT_ORDER:
            r = get(a, o)
            if r is None:
                va.append("--")
                continue
            v = pct(r["val_acc"])
            if r["val_acc"] == best_val:
                v = r"\textbf{" + v + "}"
            if (a, o) == best["key"]:
                v = r"\underline{" + v + "}"
            va.append(v)
        if j:
            lines.append(r"\addlinespace[2pt]")
        lines.append(f"{a} & Train & " + " & ".join(tr) + r" \\")
        lines.append(f" & Val & " + " & ".join(va) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(tab_dir, "accuracy.tex"), "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--report", default=os.path.join(HERE, "..", "report"))
    a = ap.parse_args()
    fig_dir, tab_dir = os.path.join(a.report, "figures"), os.path.join(a.report, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    runs = load_runs(a.results)
    d = load_data()
    digits = [int(x) for x in d["digits"]]
    archs = [x for x in ARCHITECTURES if any((x, o) in runs for o in OPT_ORDER)]

    # the initial loss must be identical for all optimizers of one architecture
    for arch in archs:
        init = {round(runs[(arch, o)]["initial_loss"], 10) for o in OPT_ORDER if (arch, o) in runs}
        print(arch, "initial losses:", init)
        assert len(init) == 1, f"{arch}: optimizers do not share the initial weights"

    # best architecture = highest validation accuracy over all (architecture, optimizer)
    key = max(runs, key=lambda k: (runs[k]["val_acc"], -runs[k]["epochs"]))
    r = runs[key]
    layout = fcnn.Layout(layer_sizes(key[0]))
    pred_tr = fcnn.predict(layout, r["theta"], d["X_train"])
    pred_te = fcnn.predict(layout, r["theta"], d["X_test"])
    cm_tr, cm_te = confusion(d["y_train"], pred_tr, 5), confusion(d["y_test"], pred_te, 5)
    best = {"key": key, "arch": key[0], "optimizer": key[1], "layers": layer_sizes(key[0]),
            "epochs": r["epochs"], "train_acc": float((pred_tr == d["y_train"]).mean()),
            "val_acc": r["val_acc"], "test_acc": float((pred_te == d["y_test"]).mean()),
            "cm_train": cm_tr.tolist(), "cm_test": cm_te.tolist(), "digits": digits}

    max_ep = max(s["epochs"] for s in runs.values())
    for arch in archs:
        plot_architecture(runs, arch, max_ep, fig_dir)
    plot_confusions([(f"Training set ({pct(best['train_acc'])}%)", cm_tr),
                     (f"Test set ({pct(best['test_acc'])}%)", cm_te)], digits, fig_dir)
    write_tables(runs, best, tab_dir, len(d["y_train"]))

    summary = {f"{k[0]}_{k[1]}": {x: v for x, v in s.items() if x not in ("hist", "theta")}
               for k, s in runs.items()}
    best_out = {k: v for k, v in best.items() if k != "key"}
    with open(os.path.join(a.results, "summary.json"), "w") as f:
        json.dump({"runs": summary, "best": best_out}, f, indent=2)
    print("BEST", {k: v for k, v in best_out.items() if not k.startswith("cm")})
    for k, s in runs.items():
        print(f"{k[0]} {k[1]:>9} epochs={s['epochs']:5d} conv={s['converged']} "
              f"E0={s['initial_loss']:.4f} E={s['final_loss']:.5f} "
              f"train={s['train_acc']:.4f} val={s['val_acc']:.4f} t={s['seconds']:.0f}s")


if __name__ == "__main__":
    main()
