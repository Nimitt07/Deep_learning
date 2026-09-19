CS601T Deep Learning - Programming Assignment 3 - Group 12 (Nimitt Jain, Dattaraj Saudagar, Onkar Mulje)
Optimizers for back-propagation in a fully connected neural network (FCNN)
==========================================================================

Files
  data_loader.py  loads the Group-12 MNIST subset (digits 0,1,2,3,7), flattens each
                  28x28 image to a 784-d vector scaled to [0,1] (cached in data_cache.npz)
  fcnn.py         FCNN, back-propagation, cross-entropy loss and the seven optimizers,
                  implemented from scratch (NumPy + Numba JIT, no DL library)
  train.py        trains every architecture with every optimizer (parallel processes)
                  and saves weights, per-epoch history and a summary per run in results/
  analyze.py      per-architecture figures of training AND validation error vs. epochs
                  (all optimizers superimposed + one panel per optimizer),
                  epoch/accuracy/final-error tables, best-model confusion matrices
  results/        <arch>_<optimizer>.json (final numbers) and .log (per-epoch log) of the
                  42 runs reported; summary.json holds all runs and the best model.
                  The .npz files (weights + per-epoch history, ~18 MB) are not included
                  to keep the upload small; train.py regenerates them.
  results/archive_eps_outside_sqrt/  logs of the first AdaGrad/RMSProp/Adam runs on
                  A1-A3 with eps outside the square root (discussed in the report)

Requirements  (pip install -r requirements.txt)
  Python 3.11, numpy, numba, matplotlib, pillow

Usage
  1. Place the dataset folder so that ../Group_12/Group_12/{train,val,test}/<digit>/ exists
     (or edit DATA_ROOT in data_loader.py).
  2. python train.py                 # all 6 architectures x 7 optimizers
     python train.py --archs A1 --opts SGD Adam --workers 2      # subset
  3. python analyze.py               # figures + LaTeX tables into ../report/

Experimental protocol (identical for every optimizer)
  * Architectures (wide / narrow network for 3, 4 and 5 hidden layers):
      A1 784-128-64-32-5          A4 784-64-32-16-5          (3 hidden layers)
      A2 784-256-128-64-32-5      A5 784-64-64-32-16-5       (4 hidden layers)
      A3 784-128-96-64-48-32-5    A6 784-64-48-32-24-16-5    (5 hidden layers)
    ReLU hidden units, softmax output, mean cross-entropy loss.
  * One He-normal initial weight vector per architecture (fixed seed) shared by all
    optimizers -> the epoch-0 loss is the same for all of them.
  * batch_size = 1 optimizers use the same per-epoch shuffled sample order.
  * Average training error E(t) = mean cross-entropy over the whole training set after
    epoch t (E(0) = before any update).  Training stops when |E(t) - E(t-1)| < 1e-4;
    max. 10000 epochs is only a safety cap (never reached).
  * lr = 0.001 for all; momentum 0.9 (GDR and NAG); RMSProp beta = 0.99, eps = 1e-8;
    Adam beta1 = 0.9, beta2 = 0.999, eps = 1e-8; AdaGrad eps = 1e-8.
  * update rules follow the course slides exactly (Class 11-14): GDR and NAG in the
    delta-w form, and eps INSIDE the square root for AdaGrad / RMSProp / Adam,
    i.e. w <- w - lr / sqrt(v + eps) * g.
