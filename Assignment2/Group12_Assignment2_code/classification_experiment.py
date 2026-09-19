"""
Shared driver for the two FCNN classification experiments (LS and NLS datasets).

For a given dataset it:
  1. loads the fixed 60 / 20 / 20 train / validation / test split,
  2. standardises the inputs using training-set statistics only,
  3. trains an FCNN with SGD backpropagation for every candidate architecture,
  4. reports the confusion matrix, accuracy, per-class and mean precision / recall /
     F-measure on the VALIDATION set for every architecture,
  5. selects the best architecture on validation accuracy,
  6. for the best architecture writes: the average-error-vs-epochs curve, the decision region
     superimposed on the training data, the outputs of every hidden and output node for the
     training / validation / test data, and the full metric report on the TEST set,
  7. trains the Assignment-1 single-neuron (one-against-one perceptron) baseline on the same
     split and writes a side-by-side comparison.
"""

import json
import os
import time

import numpy as np

import metrics as M
import plotting as P
from data_prep import Standardizer
from fcnn import FCNN, one_hot
from baseline_single_neuron import train_ovo_perceptrons, predict_ovo


def _arch_label(hidden):
    return "-".join(str(h) for h in hidden)


def run_classification_experiment(dataset_title, loader, architectures, out_dir,
                                  learning_rate=0.05, epochs=200, patience=25,
                                  hidden_activation="logistic", seed=0,
                                  baseline_lr=0.05, baseline_epochs=300):
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    X_train, y_train, X_val, y_val, X_test, y_test = loader()
    classes = sorted(set(y_train.tolist()))
    n_in, n_out = X_train.shape[1], len(classes)

    # Inputs are z-scored with training-set statistics only (see Standardizer docstring).
    scaler = Standardizer(X_train)
    Xtr, Xva, Xte = (scaler.transform(X) for X in (X_train, X_val, X_test))
    Ytr = one_hot(y_train, classes, "logistic")
    Yva = one_hot(y_val, classes, "logistic")

    print("=" * 78)
    print(f"{dataset_title}  --  FCNN classification")
    print("=" * 78)
    print(f"train = {X_train.shape[0]}, validation = {X_val.shape[0]}, "
          f"test = {X_test.shape[0]}, classes = {classes}")
    print()

    # ------------------------------------------------------------------
    # 3-4) Train every candidate architecture and score it on validation
    # ------------------------------------------------------------------
    results = []
    selection_lines = [
        f"{dataset_title} - FCNN model selection",
        "Validation-set results for every architecture that was tried.",
        "(hidden activation: " + hidden_activation +
        ", output activation: logistic, loss: squared error, learning: SGD backpropagation)",
        "Selection criterion: highest validation classification accuracy; if several",
        "architectures tie, the one with the smallest validation average error E_av is kept.",
        "",
    ]

    for idx, hidden in enumerate(architectures):
        layer_sizes = [n_in] + list(hidden) + [n_out]
        net = FCNN(layer_sizes, hidden_activation, "logistic", learning_rate, seed=seed + idx)
        t0 = time.time()
        history = net.train(Xtr, Ytr, Xva, Yva, epochs=epochs, patience=patience)

        pred_tr = net.predict_class(Xtr, classes)
        pred_va = net.predict_class(Xva, classes)
        acc_tr = M.accuracy(y_train.tolist(), pred_tr.tolist())
        report_va, stats_va = M.classification_report_text(
            y_val.tolist(), pred_va.tolist(), classes,
            f"Architecture {net.architecture_string()}  (hidden nodes: {_arch_label(hidden)})"
            "  --  VALIDATION data")

        results.append({
            "hidden": list(hidden),
            "label": _arch_label(hidden),
            "arch": net.architecture_string(),
            "net": net,
            "history": history,
            "train_accuracy": acc_tr,
            "val_stats": stats_va,
            "val_error": net.average_error(Xva, Yva),
        })

        selection_lines.append(report_va)
        selection_lines.append(
            f"Training accuracy: {acc_tr * 100:.2f}%   |   epochs run: {history['epochs_run']}"
            f"   |   parameters: {net.n_parameters()}")
        selection_lines.append("")
        print(f"  [{idx + 1}/{len(architectures)}] {net.architecture_string():<12} "
              f"train acc = {acc_tr * 100:6.2f}%   val acc = "
              f"{stats_va['accuracy'] * 100:6.2f}%   epochs = {history['epochs_run']:3d}"
              f"   ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # 5) Select the best architecture on the validation set
    # ------------------------------------------------------------------
    best = max(results, key=lambda r: (r["val_stats"]["accuracy"], -r["val_error"]))
    print(f"\n  Best architecture on validation: {best['arch']}  "
          f"(val accuracy {best['val_stats']['accuracy'] * 100:.2f}%)\n")

    selection_lines.append("=" * 78)
    selection_lines.append(f"BEST ARCHITECTURE (highest validation accuracy): {best['arch']}")
    selection_lines.append("=" * 78)

    # comparison table across architectures
    table = ["", "Summary across architectures (validation set)", "-" * 78,
             f"{'Hidden nodes':<16}{'Arch':<14}{'TrainAcc%':<12}{'ValAcc%':<12}"
             f"{'MeanPrec':<11}{'MeanRec':<11}{'MeanF':<10}"]
    for r in results:
        s = r["val_stats"]
        mark = "  <-- best" if r is best else ""
        table.append(f"{r['label']:<16}{r['arch']:<14}{r['train_accuracy'] * 100:<12.2f}"
                     f"{s['accuracy'] * 100:<12.2f}{s['mean_precision']:<11.4f}"
                     f"{s['mean_recall']:<11.4f}{s['mean_f_measure']:<10.4f}{mark}")
    selection_lines.extend(table)

    with open(os.path.join(out_dir, "model_selection_validation.txt"), "w") as f:
        f.write("\n".join(selection_lines))

    with open(os.path.join(out_dir, "results_table.csv"), "w") as f:
        f.write("hidden_nodes,architecture,train_accuracy,val_accuracy,"
                "val_mean_precision,val_mean_recall,val_mean_f_measure,epochs_run,parameters\n")
        for r in results:
            s = r["val_stats"]
            f.write(f"{r['label']},{r['arch']},{r['train_accuracy']:.4f},{s['accuracy']:.4f},"
                    f"{s['mean_precision']:.4f},{s['mean_recall']:.4f},"
                    f"{s['mean_f_measure']:.4f},{r['history']['epochs_run']},"
                    f"{r['net'].n_parameters()}\n")

    P.plot_model_selection(
        [r["label"] for r in results],
        [r["train_accuracy"] * 100 for r in results],
        [r["val_stats"]["accuracy"] * 100 for r in results],
        "Classification accuracy (%)",
        f"{dataset_title}\nAccuracy vs number of hidden nodes",
        os.path.join(out_dir, "model_selection_accuracy.png"))

    # one error-vs-epochs figure per architecture, in <out_dir>/error_curves/
    P.save_error_curve_per_architecture(
        [{"label": r["label"], "arch": r["arch"],
          "train_error_curve": r["history"]["train_error"],
          "val_error_curve": r["history"]["val_error"]} for r in results],
        out_dir, dataset_title, learning_rate=learning_rate, total_epochs=epochs)

    # ------------------------------------------------------------------
    # 6) Everything that is asked for the BEST architecture only
    # ------------------------------------------------------------------
    net = best["net"]
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

    # (2) decision region superimposed on the training data (and, for reference, val/test)
    def predict_labels(X_raw):
        return net.predict_class(scaler.transform(X_raw), classes)

    for split_name, Xs, ys in (("train", X_train, y_train),
                               ("val", X_val, y_val),
                               ("test", X_test, y_test)):
        P.plot_decision_regions(
            predict_labels, Xs, ys, classes,
            f"{dataset_title}\nDecision regions of FCNN {net.architecture_string()} "
            f"superimposed on {split_name} data",
            os.path.join(best_dir, f"decision_region_{split_name}.png"))

    # (4) outputs of every hidden node and output node, for train / val / test
    # one figure per node, under best_architecture/node_outputs/<layer>/<split>/
    node_counts = P.save_all_node_output_plots(
        net,
        [("train", X_train), ("val", X_val), ("test", X_test)],
        best_dir,
        f"{dataset_title} - FCNN {net.architecture_string()}",
        input_dim=n_in,
        transform=scaler.transform)

    # (3) full metric report on the test data for the best architecture
    pred_te = net.predict_class(Xte, classes)
    report_te, stats_te = M.classification_report_text(
        y_test.tolist(), pred_te.tolist(), classes,
        f"{dataset_title} - BEST architecture {net.architecture_string()} - TEST data")
    pred_tr = net.predict_class(Xtr, classes)
    report_tr, stats_tr = M.classification_report_text(
        y_train.tolist(), pred_tr.tolist(), classes,
        f"{dataset_title} - BEST architecture {net.architecture_string()} - TRAINING data")
    report_va, stats_va_best = M.classification_report_text(
        y_val.tolist(), net.predict_class(Xva, classes).tolist(), classes,
        f"{dataset_title} - BEST architecture {net.architecture_string()} - VALIDATION data")

    with open(os.path.join(best_dir, "best_architecture_report.txt"), "w") as f:
        f.write("\n\n".join([report_tr, report_va, report_te]))
        f.write(f"\n\nEpochs run: {best['history']['epochs_run']}   "
                f"Learning rate: {learning_rate}   "
                f"Hidden activation: {hidden_activation}   Output activation: logistic\n")
    print(report_te)
    print()

    # ------------------------------------------------------------------
    # 7) Comparison with the Assignment-1 single-neuron model
    # ------------------------------------------------------------------
    clf, _curves = train_ovo_perceptrons(Xtr, y_train, activation="logistic",
                                         learning_rate=baseline_lr,
                                         epochs=baseline_epochs, seed=seed + 50)
    base_va = predict_ovo(clf, Xva, "logistic")
    base_te = predict_ovo(clf, Xte, "logistic")
    base_report_va, base_stats_va = M.classification_report_text(
        y_val.tolist(), base_va.tolist(), classes,
        f"{dataset_title} - Assignment-1 single neuron (one-against-one) - VALIDATION data")
    base_report_te, base_stats_te = M.classification_report_text(
        y_test.tolist(), base_te.tolist(), classes,
        f"{dataset_title} - Assignment-1 single neuron (one-against-one) - TEST data")

    P.plot_decision_regions(
        lambda X_raw: predict_ovo(clf, scaler.transform(X_raw), "logistic"),
        X_train, y_train, classes,
        f"{dataset_title}\nAssignment-1 single neuron (one-against-one) - training data",
        os.path.join(out_dir, "baseline_single_neuron_decision_region_train.png"))

    cmp_lines = [
        f"{dataset_title} - FCNN (Assignment-2) vs single neuron (Assignment-1)",
        "Both models are trained and evaluated on the same 60/20/20 split with the same",
        "from-scratch metric code, so the numbers are directly comparable.",
        "",
        f"{'Model':<42}{'ValAcc%':<11}{'TestAcc%':<11}{'TestMeanP':<12}"
        f"{'TestMeanR':<12}{'TestMeanF':<11}",
        "-" * 99,
        f"{'Single neuron (one-against-one)':<42}"
        f"{base_stats_va['accuracy'] * 100:<11.2f}{base_stats_te['accuracy'] * 100:<11.2f}"
        f"{base_stats_te['mean_precision']:<12.4f}{base_stats_te['mean_recall']:<12.4f}"
        f"{base_stats_te['mean_f_measure']:<11.4f}",
        f"{('FCNN ' + net.architecture_string()):<42}"
        f"{stats_va_best['accuracy'] * 100:<11.2f}{stats_te['accuracy'] * 100:<11.2f}"
        f"{stats_te['mean_precision']:<12.4f}{stats_te['mean_recall']:<12.4f}"
        f"{stats_te['mean_f_measure']:<11.4f}",
        "",
        f"Absolute gain of the FCNN on test accuracy: "
        f"{(stats_te['accuracy'] - base_stats_te['accuracy']) * 100:+.2f} percentage points",
        "",
        base_report_va,
        "",
        base_report_te,
    ]
    with open(os.path.join(out_dir, "comparison_with_assignment1.txt"), "w") as f:
        f.write("\n".join(cmp_lines))
    print("\n".join(cmp_lines[:10]))

    # ------------------------------------------------------------------
    # Structured dump of every number, so the report can be assembled
    # without re-training anything.
    # ------------------------------------------------------------------
    summary = {
        "task": "classification",
        "dataset_title": dataset_title,
        "classes": classes,
        "learning_rate": learning_rate,
        "max_epochs": epochs,
        "hidden_activation": hidden_activation,
        "output_activation": "logistic",
        "split_sizes": {"train": int(X_train.shape[0]), "val": int(X_val.shape[0]),
                        "test": int(X_test.shape[0])},
        "architectures": [
            {
                "hidden": r["hidden"],
                "label": r["label"],
                "arch": r["arch"],
                "parameters": r["net"].n_parameters(),
                "epochs_run": r["history"]["epochs_run"],
                "train_accuracy": r["train_accuracy"],
                "val": r["val_stats"],
                "train_error_curve": r["history"]["train_error"],
                "val_error_curve": r["history"]["val_error"],
                "is_best": r is best,
            }
            for r in results
        ],
        "best": {
            "arch": net.architecture_string(),
            "hidden": best["hidden"],
            "epochs_run": best["history"]["epochs_run"],
            "node_counts": node_counts,
            "train": stats_tr,
            "val": stats_va_best,
            "test": stats_te,
            "train_error_curve": best["history"]["train_error"],
            "val_error_curve": best["history"]["val_error"],
        },
        "baseline_single_neuron": {"val": base_stats_va, "test": base_stats_te},
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  outputs written to: {out_dir}")
    print(f"  total time: {time.time() - t_start:.1f}s\n")

    return {
        "dataset_title": dataset_title,
        "results": results,
        "best": best,
        "test_stats": stats_te,
        "baseline_test_stats": base_stats_te,
        "baseline_val_stats": base_stats_va,
    }
