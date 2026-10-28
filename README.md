# ZEKAD

ZEKAD is an unsupervised anomaly detection method that combines
Zentropy-based structural modeling with K-nearest neighbor (KNN)
and local outlier factor (LOF) density analysis.

The method builds a hierarchical granular-ball representation,
models support relationships between feature subspaces and their
complements, and uses rank-based fusion to combine structural
and density scores.

## Main Features

- Hierarchical granular-ball representation
- Strictly nested Zentropy-based structural modeling
- Multi-configuration structural score consensus
- KNN and LOF density analysis
- Tie-aware mid-rank calibration and score fusion

## Requirements

- Python 3.8 or later
- NumPy
- SciPy
- scikit-learn
- PyOD

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Input Data

The code expects MATLAB `.mat` files containing a two-dimensional
numerical matrix named `trandata`. The last column contains binary
labels, where `1` denotes an anomaly, and the remaining columns
contain features.

**Datasets are not included in this repository.** Users must
prepare their own data in the required format.

Labels are used for evaluation metrics only and should not be used
to construct or orient anomaly scores.

## Running the Algorithm

First, run the structural branch:

```bash
python runner.py \
  --data-dir "/path/to/your/data" \
  --output-dir "/path/to/structural_results" \
  --expected-datasets 24
```

Then, run the density and fusion branch:

```bash
python runner_improved.py \
  --data-dir "/path/to/your/data" \
  --base-run-dir "/path/to/structural_results" \
  --output-dir "/path/to/final_results" \
  --expected-datasets 24
```

Replace the example paths with your actual directories.
The input directory must contain the expected number of valid
MATLAB files. The structural branch must finish successfully
before the density and fusion branch can be run.

The default parameter grids are defined in the source code.
Check the command-line options before changing the experimental
configuration.

## Reproducibility

The source code provides the structural modeling, density scoring,
and score fusion procedures. Exact reproduction of reported
experimental results may also depend on the dataset versions,
preprocessing, parameter settings, and evaluation protocol.

## Citation

If you use this code in your research, please cite the corresponding
ZEKAD paper. Full bibliographic information will be added when
the publication details are available.
