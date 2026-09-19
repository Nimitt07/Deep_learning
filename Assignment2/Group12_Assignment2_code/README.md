# CS601T Deep Learning — Programming Assignment II (Group 12)

Fully connected neural networks (FCNN) implemented **from scratch** and trained with
**stochastic gradient descent backpropagation** using the **squared error** as the
instantaneous loss function. No TensorFlow / PyTorch / scikit-learn / autograd is used —
`numpy` appears only for array arithmetic and `matplotlib` only for rendering figures, exactly
as in Assignment-1.

## Tasks covered

| Task | Dataset | Model required by the assignment | Script |
|---|---|---|---|
| Classification 1 | LS (linearly separable, 3 classes, 2-D) | FCNN, **one** hidden layer | `classification_ls.py` |
| Classification 2 | NLS (nonlinearly separable, 3 classes, 2-D) | FCNN, **two** hidden layers | `classification_nls.py` |
| Regression 1 | Univariate (1-D input) | FCNN, **one** hidden layer | `regression_univariate.py` |
| Regression 2 | Bivariate (2-D input) | FCNN, **one and two** hidden layers | `regression_bivariate.py` |

Several hidden-node counts are tried for every dataset; the best architecture is chosen on the
**validation** set and only that architecture is used for the plots that the assignment asks
for "for the best architecture".

### Architectures compared

Every hidden-node count is a **power of two, up to 64**. Where there are two hidden layers the
**first is always wider than the second**, so the network narrows towards the output.

| Task | Hidden layers tried | Total hidden nodes |
|---|---|---|
| LS classification | `[8]`, `[16]`, `[32]`, `[64]` | 8 – 64 |
| NLS classification | `[16,8]`, `[32,16]`, `[64,16]`, `[64,32]` | 24 – 96 |
| Univariate regression | `[8]`, `[16]`, `[32]`, `[64]` | 8 – 64 |
| Bivariate regression | `[16]`, `[32]`, `[64]`, `[32,16]`, `[64,32]` | 16 – 96 |

### Error-vs-epochs plots and early stopping

Training often stops before the epoch budget is spent (the stopping criterion fires, or the
validation error stops improving). **The curve simply ends there.** After the stopping epoch
no training happens, so there is no average error to plot; drawing a continuation would
suggest an error had been measured while training carried on, which is not what happened.

* `error_vs_epochs.png` — the x-axis ends at the last trained epoch, the title says
  `stopped early after N of M epochs` when the budget was not used up, and a ring marks the
  epoch whose weights were restored (the lowest validation error).
* `error_curves/hidden_<label>.png` — **one graph per architecture**, not all of them
  overlaid on a single axis with a legend. Overlaid curves sit on top of one another and
  none of them can be read; a separate graph per architecture shows each one properly, with
  its own training and validation curve.

### Node-output plots: one figure per node

The assignment asks for the output of **each** hidden node and output node. With 50-100 nodes
per layer a shared grid figure is unreadable, so **every node gets its own PNG**, filed under
`outputs/<experiment>/best_architecture/node_outputs/<layer>/<split>/node_NNN.png`. The PDF
report reproduces the first few nodes of each layer and points at the complete set on disk.

This produces roughly **620 PNG files (~44 MB)** in total, because a 96-node network needs
96 figures for each of the three splits.

> **If this folder lives inside OneDrive (or Dropbox / Google Drive), run the experiments
> somewhere else.** The sync client tries to upload each PNG as it is written and competes
> with training for the CPU: on this machine one architecture that trains in 65 s took 51
> minutes with OneDrive syncing in the background. Either copy the code to a local folder
> (e.g. `C:\Temp\`), run it there and copy `outputs/` back afterwards, or pause syncing while
> the run is in progress.

## How to run

```bash
python run_all.py
```

or one experiment at a time:

```bash
python data_prep.py             # builds/prints the 60/20/20 splits
python classification_ls.py
python classification_nls.py
python regression_univariate.py
python regression_bivariate.py
```

and, optionally,

```bash
python test_gradient_check.py     # verifies backpropagation against finite differences
python compare_initialisation.py  # measures the four weight-initialisation schemes
```

Requirements: Python 3.8+, `numpy` and `matplotlib`. Nothing else.

The report itself (`Group12_Assignment2_report.pdf`) is a separate deliverable and is not
built from this folder; the scripts here produce the results and figures it is made from.

### On Google Colab

Open **`Run_on_Colab.ipynb`** in Colab and run the cells top to bottom, or do it by hand:

```python
from google.colab import files            # then pick Group12_Assignment2_code.zip
files.upload()

!unzip -q -o Group12_Assignment2_code.zip
%cd Group12_Assignment2_code

!python test_gradient_check.py            # 5 s sanity check
!python -u run_all.py                     # ~20 min on a CPU runtime

!zip -q -r outputs.zip outputs
files.download('outputs.zip')
```

`numpy` and `matplotlib` are pre-installed on Colab, so there is nothing to `pip install`.

* **Keep the runtime on CPU.** The network is plain numpy, so a GPU runtime is no faster.
* The zip already contains the raw `Group12/` data and the cached `processed_data/` splits,
  so nothing else needs uploading.
* The whole run takes roughly 25 minutes; Colab drops an *idle* notebook after about 90
  minutes, so leave the tab open. If the runtime keeps dropping, run the four experiment
  scripts separately and then `python run_all.py --summary-only`.
* `outputs/` ends up around 45 MB, so zip it before downloading (`!zip -r outputs.zip
  outputs`) or copy it to Google Drive.

## Data

The raw data folder `Group12/` (the one issued to Group 12 for Assignment-1) is looked up
automatically in `./Group12`, `../Group12` or `../../Group12`, so the code runs from the
submission zip or from the lab folder without editing paths.

Each dataset is split **once**, per class for the classification datasets, into
**60 % train / 20 % validation / 20 % test** with a fixed seed, and cached in
`processed_data/*.npz`. Every script loads the same cached split, so changing the architecture
never reshuffles the data. Delete the `.npz` file to force a fresh split.

| Dataset | Train | Validation | Test |
|---|---|---|---|
| LS classification | 900 (300/class) | 300 (100/class) | 300 (100/class) |
| NLS classification | 900 (300/class) | 300 (100/class) | 300 (100/class) |
| Univariate regression | 601 | 200 | 200 |
| Bivariate regression | 6121 | 2040 | 2040 |

## Files

| File | Contents |
|---|---|
| `fcnn.py` | The FCNN: forward pass, backpropagation, pattern-by-pattern SGD, activations, 1-of-K coding. **This is the core of the assignment.** |
| `metrics.py` | Confusion matrix, accuracy, per-class/mean precision, recall, F-measure, RMSE, %RMSE — all written from scratch. |
| `data_prep.py` | Loading, the fixed 60/20/20 split, and the `Standardizer`. |
| `plotting.py` | Every figure: error curves, decision regions, hidden/output-node surfaces, model-vs-target, target-vs-model scatter. |
| `perceptron.py` | The Assignment-1 single neuron (unchanged), used as the comparison baseline. |
| `baseline_single_neuron.py` | Assignment-1 models re-run on this split: one-against-one perceptrons and the single linear neuron. |
| `classification_experiment.py` | Driver shared by the two classification scripts. |
| `regression_experiment.py` | Driver shared by the two regression scripts. |
| `run_all.py` | Runs everything and writes `outputs/SUMMARY.txt`. |
| `test_gradient_check.py` | Verifies the backpropagation against central finite differences. |
| `compare_initialisation.py` | Measures the four weight-initialisation schemes side by side; picks the default. |
| `Run_on_Colab.ipynb` | Ready-to-run Google Colab notebook (upload the zip, run the cells). |

### Correctness of the backpropagation

`python test_gradient_check.py` compares the analytic weight change produced by one SGD step
against a central finite-difference estimate of the same derivative of `E_n`. All six
configurations tested (one, two and three hidden layers; logistic and tanh hidden units;
logistic and linear output units) agree to a relative error of about `1e-11`, so the forward
pass, the local gradients and the weight update are consistent.

## Notation

`fcnn.py` follows the notation of the course slides (*Class07-10: Basics of Neural Networks /
FCNN*, slides 32–36) so the code can be read side by side with the lecture.

| Symbol | Meaning | In the code |
|---|---|---|
| `D = {(x_n, y_n)}`, `x_n ∈ R^d`, `y_n ∈ R^K` | training data | `X`, `Y` |
| `d`, `J`, `K` | input / hidden / output node counts (indices `i`, `j`, `k`) | `layer_sizes` |
| `W^(h)`, entries `w_ij^(h)` | input → hidden weights | `net.W[0][i][j]` |
| `W^(o)`, entries `w_jk^(o)` | hidden → output weights | `net.W[-1][j][k]` |
| `a_nj`, `h_nj = g(a_nj)` | hidden node net input / output | `outputs[l]` |
| `a_nk^(o)`, `ŷ_nk = f(a_nk^(o))` | output node net input / output | `yhat_n` |
| `g(·)` / `f(·)` | hidden / output activation | `ACTIVATIONS` |
| `η` | learning rate | `net.eta` |
| `δ_nk^(o)`, `δ_nj^(h)` | local gradients | `delta[-1]`, `delta[l]` |
| `E_n`, `E_av` | instantaneous / average error | `E_n`, `E_av` |

The bias is the weight of **index 0**, exactly as in the slides' sums that begin at `j = 0`
(`a_k^(o) = Σ_{j=0}^J w_jk^(o) h_j` with `h_0 = 1`). Every weight matrix is therefore stored
with the **from**-index as its row and the **to**-index as its column — `W[0][i][j]` *is*
`w_ij^(h)` — and row 0 holds the bias.

## The algorithm, step by step (slide 34)

`FCNN.train` and `FCNN._update_on_one_example` implement these seven steps one-for-one:

1. Initialise `W^(h)` and `W^(o)` with random values.
2. Choose a training example `x_n`.
3. **Forward computation** — compute the output of all output neurons `ŷ_nk`.
4. **Backward operation** — `E_n = ½ Σ_k (y_nk − ŷ_nk)²`, then
   `δ_nk^(o) = (y_nk − ŷ_nk) ∂f(a_nk^(o))/∂a_nk^(o)`, `Δw_jk^(o) = η δ_nk^(o) h_nj`, and
   `δ_nj^(h) = [Σ_k δ_nk^(o) w_jk^(o)] ∂g(a_nj)/∂a_nj`, `Δw_ij^(h) = η δ_nj^(h) x_i`.
5. Repeat 2–4 until every training example has been presented once — one **epoch**.
6. Compute the average error `E_av = (1/N) Σ_n E_n` (this is the curve that is plotted).
7. Repeat 2–6 until the stopping criterion is satisfied.

With two hidden layers, step 4's hidden-layer rule is applied again layer by layer, the sum
over `k` being taken over the nodes of the layer immediately above.

## Model details

* **Hidden activation** `g(a_j) = 1/(1 + e^{-a_j})`, `∂g/∂a = g(1 − g)`; `tanh` also supported.
* **Output activation** `f` — logistic for classification (1-of-K coded targets), linear for
  regression.
* **Test phase** (slide 36) — classification: class label for `x` = `argmax_k ŷ_k`
  (`FCNN.predict_class`); regression: `ŷ_k = a_k^(o)` (`FCNN.predict_value`).
* **Initialisation** — all schemes centre the weights on 0 and set the spread from the layer
  size, so the initial net inputs stay in the active region of the logistic function:

  | `init=` | Distribution | |
  |---|---|---|
  | `uniform2` | `U(−2/√n, +2/√n)` | **default** (mean ± 2σ) |
  | `uniform` | `U(−1/√n, +1/√n)` | mean ± 1σ |
  | `normal` | `N(0,1)/√n` | the rule given on slide 35 |
  | `xavier` | `U(−√(6/(n_in+n_out)), +…)` | Glorot uniform |

  The default was **chosen by measurement, not assumption**. `python compare_initialisation.py`
  trains all four datasets with all four schemes — same architecture, learning rate, epoch
  budget, averaged over three seeds — and `uniform2` came out best both on how quickly `E_av`
  descends (mean rank 1.75 of 4) and on the final validation metric (mean rank 2.13 of 4).
  Widening the range from ±1σ to ±2σ is what matters: at ±1σ the net inputs are so small that
  every logistic unit starts in the near-linear part of its curve and the hidden units take
  many epochs to differentiate.
* **Stopping criterion** (slide 35) — a fixed number of epochs is reached, **or**
  `|E_av(t) − E_av(t−1)|` falls below a threshold. Since the weights are updated after every
  single example the `E_av` curve is noisy, so the condition must hold for `patience`
  consecutive epochs and the threshold is `1e-8` rather than the slide's example `1e-3` —
  with a single flat epoch, or the larger threshold, training halts on noise long before it
  has converged. Additionally (not part of the slide) training stops when the validation
  error has not improved for `patience` epochs, and the weights of the best validation epoch
  are restored.
* **Input normalisation** (slide 35) — inputs, and for regression the targets, are z-scored
  using **training-set statistics only**. The logistic units saturate on the raw LS inputs
  (`x₁ ≈ 20`) and the raw bivariate targets (up to ≈ 90), and SGD then barely moves. All
  reported RMSE / %RMSE values are computed after inverse-transforming the predictions back to
  the original units, so they stay directly comparable with the Assignment-1 numbers.

## Outputs

```
outputs/
  SUMMARY.txt                                  headline numbers for all four experiments
  classification_ls/  classification_nls/
      model_selection_validation.txt           confusion matrix + accuracy + precision/recall/
                                               F-measure (per class and mean) on VALIDATION
                                               data for EVERY architecture tried
      results_table.csv                        the same, as a table
      model_selection_accuracy.png
      error_curves/hidden_<label>.png           ONE ERROR GRAPH PER ARCHITECTURE
      comparison_with_assignment1.txt          FCNN vs Assignment-1 single neuron
      baseline_single_neuron_decision_region_train.png
      results.json                             every number above, machine-readable
                                               (every number, for the report)
      best_architecture/
          best_model.npz                       trained weights (FCNN.load reloads them)
          error_vs_epochs.png                  (result 1)
          decision_region_train/val/test.png   (result 2)
          best_architecture_report.txt         (result 3: train, validation and TEST metrics)
          node_outputs/                        (result 4) ONE FIGURE PER NODE
              hidden_layer1/{train,val,test}/node_001.png, node_002.png, ...
              hidden_layer2/{train,val,test}/node_001.png, ...
              output_layer/{train,val,test}/node_001.png, ...
  regression_univariate/  regression_bivariate/
      model_selection_and_best.txt             RMSE and %RMSE on training and validation data
                                               for every architecture + test values for the best
      results_table.csv
      results.json
      model_selection_rmse.png
      error_curves/hidden_<label>.png           ONE ERROR GRAPH PER ARCHITECTURE
      comparison_with_assignment1.txt
      baseline_single_neuron_fit_train.png
      best_architecture/
          best_model.npz                             trained weights
          error_vs_epochs.png                        (result 1)
          model_vs_target_train/val/test.png         (result 3)
          scatter_target_vs_model_train/val/test.png (result 4)
          node_outputs/                              (result 5) ONE FIGURE PER NODE
              hidden_layer1/{train,val,test}/node_001.png, node_002.png, ...
              hidden_layer2/{train,val,test}/node_001.png, ...
              output_layer/{train,val,test}/node_001.png
```

## Submission

* Code folder: `Group12_Assignment2_code`
* Zip: `Group12_Assignment2_code.zip`
* Report: `Group12_Assignment2_report.pdf`
