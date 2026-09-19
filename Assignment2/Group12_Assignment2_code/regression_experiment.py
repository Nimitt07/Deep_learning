"""
Shared driver for the two FCNN regression experiments (univariate and bivariate datasets).

For a given dataset it:
  1. loads the fixed 60 / 20 / 20 train / validation / test split,
  2. standardises inputs and targets using training-set statistics only,
  3. trains an FCNN with SGD backpropagation (linear output node, squared error loss) for
     every candidate architecture,
  4. reports RMSE and %RMSE on training and validation data for every architecture,
  5. selects the best architecture on validation RMSE,
  6. for the best architecture writes: the average-error-vs-epochs curve, the model output
     superimposed on the target output, the target-vs-model scatter plot, the outputs of every
     hidden and output node (all for training / validation / test data), and the RMSE and
     %RMSE on the test data,
  7. trains the Assignment-1 single-neuron (linear) baseline on the same split and writes a
     side-by-side comparison.

All RMSE / %RMSE numbers are computed in the ORIGINAL target units: the network's output is
inverse-transformed before the metrics are evaluated.
"""

import json
import os
import time

import numpy as np

import metrics as M
import plotting as P
from data_prep import Standardizer
from fcnn import FCNN
from baseline_single_neuron import train_linear_neuron


def _arch_label(hidden):
    return "-".join(str(h) for h in hidden)


def run_regression_experiment(dataset_title, loader, architectures, out_dir,
                              learning_rate=0.01, epochs=200, patience=20,
                              hidden_activation="logistic", seed=0,
                              baseline_lr=0.01, baseline_epochs=200):
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    X_train, y_train, X_val, y_val, X_test, y_test = loader()
    n_in = X_train.shape[1]

    # z-score inputs and targets on the training split only; predictions are mapped back to
    # the original units before any metric is computed.
    sx = Standardizer(X_train)
    sy = Standardizer(y_train)
    Xtr, Xva, Xte = (sx.transform(X) for X in (X_train, X_val, X_test))
    Ytr = sy.transform(y_train).reshape(-1, 1)
    Yva = sy.transform(y_val).reshape(-1, 1)

    print("=" * 78)
    print(f"{dataset_title}  --  FCNN regression")
    print("=" * 78)
    print(f"train = {X_train.shape[0]}, validation = {X_val.shape[0]}, "
          f"test = {X_test.shape[0]}, input dimension = {n_in}")
    print()

    def predict_raw(net, X_scaled):
        """Network output mapped back to the original target units."""
        return sy.inverse(net.predict_value(X_scaled))

    # ------------------------------------------------------------------
    # 3-4) Train every candidate architecture, score it on train + validation
    # ------------------------------------------------------------------
    results = []
    for idx, hidden in enumerate(architectures):
        layer_sizes = [n_in] + list(hidden) + [1]
        net = FCNN(layer_sizes, hidden_activation, "linear", learning_rate, seed=seed + idx)
        t0 = time.time()
        history = net.train(Xtr, Ytr, Xva, Yva, epochs=epochs, patience=patience)

        p_tr = predict_raw(net, Xtr)
        p_va = predict_raw(net, Xva)
        rec = {
            "hidden": list(hidden),
            "label": _arch_label(hidden),
            "arch": net.architecture_string(),
            "n_hidden_layers": len(hidden),
            "net": net,
            "history": history,
            "rmse_train": M.rmse(y_train.tolist(), p_tr.tolist()),
            "rmse_val": M.rmse(y_val.tolist(), p_va.tolist()),
            "prmse_train": M.percent_rmse(y_train.tolist(), p_tr.tolist()),
            "prmse_val": M.percent_rmse(y_val.tolist(), p_va.tolist()),
        }
        results.append(rec)
        print(f"  [{idx + 1}/{len(architectures)}] {net.architecture_string():<14} "
              f"RMSE train = {rec['rmse_train']:9.4f}   RMSE val = {rec['rmse_val']:9.4f}   "
              f"%RMSE val = {rec['prmse_val']:7.3f}%   epochs = {history['epochs_run']:3d}"
              f"   ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # 5) Best architecture = lowest validation RMSE
    # ------------------------------------------------------------------
    best = min(results, key=lambda r: r["rmse_val"])
    net = best["net"]
    print(f"\n  Best architecture on validation: {best['arch']} "
          f"(val RMSE {best['rmse_val']:.4f})\n")

    lines = [
        f"{dataset_title} - FCNN model selection",
        "RMSE and %RMSE on training and validation data for every architecture that was tried.",
        f"(hidden activation: {hidden_activation}, output activation: linear, "
        "loss: squared error, learning: SGD backpropagation)",
        "Selection criterion: lowest validation RMSE.",
        "",
        f"{'Hidden nodes':<16}{'Arch':<16}{'#HidLayers':<12}{'RMSE(train)':<14}"
        f"{'%RMSE(train)':<15}{'RMSE(val)':<14}{'%RMSE(val)':<14}{'Epochs':<8}{'Params':<8}",
        "-" * 117,
    ]
    for r in results:
        mark = "  <-- best" if r is best else ""
        lines.append(f"{r['label']:<16}{r['arch']:<16}{r['n_hidden_layers']:<12}"
                     f"{r['rmse_train']:<14.4f}{r['prmse_train']:<15.4f}"
                     f"{r['rmse_val']:<14.4f}{r['prmse_val']:<14.4f}"
                     f"{r['history']['epochs_run']:<8}{r['net'].n_parameters():<8}{mark}")

    # ------------------------------------------------------------------
    # 6) Best architecture: test metrics + all requested plots
    # ------------------------------------------------------------------
    pred = {
        "train": (X_train, y_train, predict_raw(net, Xtr)),
        "val": (X_val, y_val, predict_raw(net, Xva)),
        "test": (X_test, y_test, predict_raw(net, Xte)),
    }
    lines += ["", "=" * 117,
              f"BEST ARCHITECTURE: {best['arch']}  (hidden nodes: {best['label']}, "
              f"{best['n_hidden_layers']} hidden layer(s), eta = {learning_rate}, "
              f"epochs run = {best['history']['epochs_run']})",
              "=" * 117,
              f"{'Split':<12}{'RMSE':<16}{'%RMSE':<16}"]
    test_metrics = {}
    for split in ("train", "val", "test"):
        _, yt, yp = pred[split]
        r = M.rmse(yt.tolist(), yp.tolist())
        pr = M.percent_rmse(yt.tolist(), yp.tolist())
        test_metrics[split] = (r, pr)
        lines.append(f"{split:<12}{r:<16.4f}{pr:<16.4f}")

    with open(os.path.join(out_dir, "model_selection_and_best.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[5:]))
    print()

    with open(os.path.join(out_dir, "results_table.csv"), "w") as f:
        f.write("hidden_nodes,architecture,n_hidden_layers,rmse_train,percent_rmse_train,"
                "rmse_val,percent_rmse_val,epochs_run,parameters\n")
        for r in results:
            f.write(f"{r['label']},{r['arch']},{r['n_hidden_layers']},{r['rmse_train']:.6f},"
                    f"{r['prmse_train']:.6f},{r['rmse_val']:.6f},{r['prmse_val']:.6f},"
                    f"{r['history']['epochs_run']},{r['net'].n_parameters()}\n")

    P.plot_model_selection(
        [r["label"] for r in results],
        [r["rmse_train"] for r in results],
        [r["rmse_val"] for r in results],
        "RMSE",
        f"{dataset_title}\nRMSE vs number of hidden nodes",
        os.path.join(out_dir, "model_selection_rmse.png"))

    # one error-vs-epochs figure per architecture, in <out_dir>/error_curves/
    P.save_error_curve_per_architecture(
        [{"label": r["label"], "arch": r["arch"],
          "train_error_curve": r["history"]["train_error"],
          "val_error_curve": r["history"]["val_error"]} for r in results],
        out_dir, dataset_title, learning_rate=learning_rate, total_epochs=epochs)

    best_dir = os.path.join(out_dir, "best_architecture")
    os.makedirs(best_dir, exist_ok=True)
    net.save(os.path.join(best_dir, "best_model.npz"))

    # (1) average error vs epochs
    P.plot_error_curve(
        best["history"],
        f"{dataset_title}\nAverage error vs epochs - best architecture "
        f"{net.architecture_string()} (eta = {learning_rate})",
        os.path.join(best_dir, "error_vs_epochs.png"),
        total_epochs=epochs)

    # (3) model output superimposed on target output, and (4) target-vs-model scatter
    for split in ("train", "val", "test"):
        Xs, yt, yp = pred[split]
        fit_title = (f"{dataset_title}\nModel output superimposed on target output "
                     f"({split} data) - FCNN {net.architecture_string()}")
        plot_fit = P.plot_fit_1d if n_in == 1 else P.plot_fit_2d
        plot_fit(Xs, yt, yp, fit_title,
                 os.path.join(best_dir, f"model_vs_target_{split}.png"))
        P.plot_target_vs_model(
            yt, yp,
            f"{dataset_title}\nTarget output vs model output "
            f"({split} data) - FCNN {net.architecture_string()}",
            os.path.join(best_dir, f"scatter_target_vs_model_{split}.png"))

    # (5) outputs of every hidden node and output node, for train / val / test
    # one figure per node, under best_architecture/node_outputs/<layer>/<split>/
    node_counts = P.save_all_node_output_plots(
        net,
        [("train", X_train), ("val", X_val), ("test", X_test)],
        best_dir,
        f"{dataset_title} - FCNN {net.architecture_string()}",
        input_dim=n_in,
        transform=sx.transform)

    # ------------------------------------------------------------------
    # 7) Comparison with the Assignment-1 single-neuron (linear) model
    # ------------------------------------------------------------------
    base, _errs = train_linear_neuron(Xtr, Ytr.ravel(), learning_rate=baseline_lr,
                                      epochs=baseline_epochs, seed=seed + 50)
    base_metrics = {}
    for split, Xs in (("train", Xtr), ("val", Xva), ("test", Xte)):
        yt = {"train": y_train, "val": y_val, "test": y_test}[split]
        yp = sy.inverse(base.predict_raw(Xs))
        base_metrics[split] = (M.rmse(yt.tolist(), yp.tolist()),
                               M.percent_rmse(yt.tolist(), yp.tolist()))

    def _cmp_row(name, table):
        cells = [f"{table[s][0]:<14.4f}" for s in ("train", "val", "test")]
        cells += [f"{table[s][1]:<14.4f}" for s in ("train", "val", "test")]
        return f"{name:<28}" + "".join(cells)

    cmp_lines = [
        f"{dataset_title} - FCNN (Assignment-2) vs single neuron (Assignment-1)",
        "Both models use the same 60/20/20 split, the same standardisation and the same",
        "from-scratch RMSE / %RMSE code, so the numbers are directly comparable.",
        "",
        f"{'Model':<28}{'RMSE(train)':<14}{'RMSE(val)':<14}{'RMSE(test)':<14}"
        f"{'%RMSE(train)':<14}{'%RMSE(val)':<14}{'%RMSE(test)':<14}",
        "-" * 112,
        _cmp_row("Single linear neuron", base_metrics),
        _cmp_row("FCNN " + net.architecture_string(), test_metrics),
        "",
        f"Reduction in test RMSE achieved by the FCNN: "
        f"{(1 - test_metrics['test'][0] / base_metrics['test'][0]) * 100:.2f}%",
    ]
    with open(os.path.join(out_dir, "comparison_with_assignment1.txt"), "w") as f:
        f.write("\n".join(cmp_lines))
    print("\n".join(cmp_lines))

    # the single-neuron fit, plotted the same way, for a like-for-like visual comparison
    yp_base_train = sy.inverse(base.predict_raw(Xtr))
    plot_fit = P.plot_fit_1d if n_in == 1 else P.plot_fit_2d
    plot_fit(X_train, y_train, yp_base_train,
             f"{dataset_title}\nAssignment-1 single linear neuron: model output "
             f"superimposed on target output (training data)",
             os.path.join(out_dir, "baseline_single_neuron_fit_train.png"))

    # ------------------------------------------------------------------
    # Structured dump of every number, so the report can be assembled
    # without re-training anything.
    # ------------------------------------------------------------------
    summary = {
        "task": "regression",
        "dataset_title": dataset_title,
        "input_dim": n_in,
        "learning_rate": learning_rate,
        "max_epochs": epochs,
        "hidden_activation": hidden_activation,
        "output_activation": "linear",
        "split_sizes": {"train": int(X_train.shape[0]), "val": int(X_val.shape[0]),
                        "test": int(X_test.shape[0])},
        "architectures": [
            {
                "hidden": r["hidden"],
                "label": r["label"],
                "arch": r["arch"],
                "n_hidden_layers": r["n_hidden_layers"],
                "parameters": r["net"].n_parameters(),
                "epochs_run": r["history"]["epochs_run"],
                "rmse_train": r["rmse_train"],
                "rmse_val": r["rmse_val"],
                "prmse_train": r["prmse_train"],
                "prmse_val": r["prmse_val"],
                "train_error_curve": r["history"]["train_error"],
                "val_error_curve": r["history"]["val_error"],
                "is_best": r is best,
            }
            for r in results
        ],
        "best": {
            "arch": net.architecture_string(),
            "hidden": best["hidden"],
            "n_hidden_layers": best["n_hidden_layers"],
            "epochs_run": best["history"]["epochs_run"],
            "node_counts": node_counts,
            "rmse": {s: test_metrics[s][0] for s in ("train", "val", "test")},
            "prmse": {s: test_metrics[s][1] for s in ("train", "val", "test")},
            "train_error_curve": best["history"]["train_error"],
            "val_error_curve": best["history"]["val_error"],
        },
        "baseline_single_neuron": {
            "rmse": {s: base_metrics[s][0] for s in ("train", "val", "test")},
            "prmse": {s: base_metrics[s][1] for s in ("train", "val", "test")},
        },
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  outputs written to: {out_dir}")
    print(f"  total time: {time.time() - t_start:.1f}s\n")

    return {
        "dataset_title": dataset_title,
        "results": results,
        "best": best,
        "fcnn_metrics": test_metrics,
        "baseline_metrics": base_metrics,
    }
