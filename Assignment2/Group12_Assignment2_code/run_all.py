"""
Runs every Assignment-2 experiment end to end and writes a combined summary.

    python run_all.py                 run all four experiments, then write outputs/SUMMARY.txt
    python run_all.py --summary-only  only rebuild outputs/SUMMARY.txt from the results already
                                      in outputs/<experiment>/results.json

Order of execution:
  1. data_prep             - builds (or reuses) the fixed 60/20/20 splits
  2. classification_ls     - FCNN, 1 hidden layer,  LS dataset
  3. classification_nls    - FCNN, 2 hidden layers, NLS dataset
  4. regression_univariate - FCNN, 1 hidden layer
  5. regression_bivariate  - FCNN, 1 and 2 hidden layers

Everything is written under outputs/<experiment>/ ; the combined summary that collects the
headline numbers for the report is outputs/SUMMARY.txt.
"""

import json
import os
import sys
import time

import data_prep

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

CLASSIFICATION = ["classification_ls", "classification_nls"]
REGRESSION = ["regression_univariate", "regression_bivariate"]


def _load(folder):
    with open(os.path.join(OUT_DIR, folder, "results.json")) as f:
        return json.load(f)


def write_summary(elapsed=None):
    """Rebuild outputs/SUMMARY.txt from the stored results.json of each experiment."""
    lines = [
        "CS601T Deep Learning - Programming Assignment II - Group 12",
        "Combined summary of all four experiments",
        "=" * 96,
        "",
        "CLASSIFICATION (accuracy and mean precision / recall / F-measure on the TEST set,",
        "for the architecture selected on the validation set)",
        "-" * 96,
        f"{'Dataset':<26}{'Best FCNN':<14}{'FCNN acc%':<12}{'1-neuron acc%':<15}"
        f"{'FCNN meanF':<12}{'1-neuron meanF':<15}",
    ]
    for folder in CLASSIFICATION:
        r = _load(folder)
        best, base = r["best"], r["baseline_single_neuron"]
        lines.append(
            f"{r['dataset_title'][:24]:<26}{best['arch']:<14}"
            f"{best['test']['accuracy'] * 100:<12.2f}"
            f"{base['test']['accuracy'] * 100:<15.2f}"
            f"{best['test']['mean_f_measure']:<12.4f}"
            f"{base['test']['mean_f_measure']:<15.4f}")

    lines += [
        "",
        "REGRESSION (RMSE and %RMSE on the TEST set, for the architecture selected on the",
        "validation set)",
        "-" * 96,
        f"{'Dataset':<26}{'Best FCNN':<14}{'FCNN RMSE':<13}{'1-neuron RMSE':<16}"
        f"{'FCNN %RMSE':<13}{'1-neuron %RMSE':<15}",
    ]
    for folder in REGRESSION:
        r = _load(folder)
        best, base = r["best"], r["baseline_single_neuron"]
        lines.append(
            f"{r['dataset_title'][:24]:<26}{best['arch']:<14}"
            f"{best['rmse']['test']:<13.4f}{base['rmse']['test']:<16.4f}"
            f"{best['prmse']['test']:<13.4f}{base['prmse']['test']:<15.4f}")

    if elapsed is not None:
        lines += ["", f"Total run time: {elapsed:.1f} s"]

    summary = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "SUMMARY.txt"), "w") as f:
        f.write(summary)
    return summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if "--summary-only" in sys.argv:
        print(write_summary())
        return

    t0 = time.time()
    print("Preparing the 60/20/20 splits ...")
    data_prep.prepare_ls_classification()
    data_prep.prepare_nls_classification()
    data_prep.prepare_univariate_regression()
    data_prep.prepare_bivariate_regression()
    try:
        print(f"Raw data directory: {data_prep.find_raw_dir()}\n")
    except FileNotFoundError:
        # fine as long as processed_data/ already holds the cached splits
        print("Raw data folder not present; using the cached splits in processed_data/.\n")

    import classification_ls
    import classification_nls
    import regression_univariate
    import regression_bivariate

    classification_ls.run()
    classification_nls.run()
    regression_univariate.run()
    regression_bivariate.run()

    print()
    print(write_summary(elapsed=time.time() - t0))


if __name__ == "__main__":
    main()
