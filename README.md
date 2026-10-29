# ZEKAD

ZEKAD is an unsupervised anomaly detector combining strictly nested Zentropy
structural modeling with KNN and LOF density scores. The implementation retains
the paper's 160-configuration structural consensus, tie-aware rank calibration,
and fixed structural weight of 0.20.

`zekad.py` contains the core algorithm; `run.py` is the MATLAB-data runner.

## Install

Use Python 3.10 or 3.11 with the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

Prepare a MATLAB `.mat` file with a numerical matrix named `trandata`: features
in all but the last column, and binary labels in the last column (`0`: normal,
`1`: anomaly; both classes required). An unlabeled feature matrix named `X` is
also accepted. Inputs must be finite and real-valued. Datasets are not included.
MATLAB v7.3 files additionally require `h5py`.

```bash
python run.py /path/to/Abalone.mat --output-dir results
python run.py /path/to/data --k 5 --beta 0 --output-dir results
```

The default is the fixed setting `(k, beta) = (5, 0)`. Each dataset produces one
NPZ containing `scores` (configurations × samples), `config_k`, `config_beta`,
and labels when present. `metrics.csv` contains one AUC/AP row per configuration;
metrics are blank for unlabeled data. Higher scores indicate greater abnormality.
Existing result files with the same names are replaced.

To generate the paper's 295 density-fusion candidates (`k = 2,...,60` and
`beta = 0,0.25,0.5,0.75,1`):

```bash
python run.py /path/to/data --grid --output-dir grid_results
```

Every run computes the same 160 structural configurations. Density neighborhoods
must satisfy `k < n`; the full grid requires at least 61 samples. Labels are used
only after score generation. Grid runs display separate label-selected AUC/AP
maxima as **oracle results**, with ties favoring smaller `k`, then smaller `beta`.

## Python API

```python
from zekad import score_samples, score_grid

scores = score_samples(X, dataset_name="Abalone", k=5, beta=0.0)
grid = score_grid(X, dataset_name="Abalone")
# grid["scores"], grid["config_k"], grid["config_beta"], grid["z_consensus"]
```

`X` contains features only. These are transductive scores for the supplied samples;
the API does not fit a model for predicting unseen samples. `dataset_name` enters
the deterministic subspace seed: preserve the original dataset file stem when
reproducing a run. The CLI uses that stem automatically.

## Paper evaluation

The main comparison uses separate AUC/AP oracle maxima from `--grid` for the
19 archived datasets. The five added datasets—Mammography, Pendigits, Satimage2,
Shuttle10k, and ThyroidDisease—use the fixed setting `(5, 0)` for both metrics.
The ablation study instead optimizes each variant on all 24 datasets; its full
model results for the five added datasets therefore use a different protocol.

The core formulas, preprocessing, seeds, and parameter grids are preserved.
Exact reproduction also requires the original dataset versions and names. The
paper's structural and density reference runs used different Python/dependency
environments; this consolidated implementation does not guarantee bitwise
agreement across versions or nearest-neighbor backends, especially for tied
distances. AUC and AP maxima are selected separately and can use different score
vectors; AP means average precision.

The refactor was checked against the previous public code on all 24 original
matrices: every structural and final score was identical under the reported
evaluation protocol. On the validation platform, 17 historical metric rows were
reproduced exactly; seven retain differences already present in the previous
code. Floating-point ties in granular-ball split seeds were confirmed as one
source of these historical differences.

## Acknowledgment

We gratefully acknowledge [Professor Mingjie Cai](https://github.com/HNU-MingjieCai),
our advisor, for his guidance and support throughout this work.

## Citation

Please cite the corresponding ZEKAD paper when using this code. Full publication
details will be added when available.
