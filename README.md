

## What it implements

The notebook uses an ECDC-style daily country time-series CSV and aligned country-level metadata. It constructs a next-day incidence-direction binary task, makes 35 candidate features where the supplied input supports them, fits imputation/scaling/`SelectKBest` on training folds only, and selects 25 features. It supports simulated country-level geographic clients plus IID and label-skew controls. The model has four hidden dense layers, each followed by batch normalization, ReLU, and dropout. Training uses AdamW for 25 rounds with 5 local epochs, a training batch size of 64, and an evaluation batch size of 128.

The implementation runs FedAvg, FedProx, IFCA, weighted/unweighted geographic clustered FL, random cluster assignments, and local-only training. It evaluates five coordinate-masking mechanisms, the declared nearest-candidate centroid attacker (AIE and Re-ID at 100 m), ARI, accuracy, macro-F1, AUC, and a loss-based MIA AUC. Per-seed records, summary confidence intervals, matched-seed paired tests, and provenance hashes are saved to `outputs/`.

## Important scope and data contract



The `planar_laplace` option is a coordinate perturbation parameterized by expected displacement. This repository does not claim a global formal geo-indistinguishability guarantee. AIE/Re-ID evaluate location-derived coordination metadata only; they do not measure privacy of model updates. 

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook GeoFedMask_reproducible_experiments.ipynb
```

Put input files at:

```text
data/raw/daily.csv
 data/raw/static_metadata.csv
```



## Generated artifacts

- `outputs/manifest.json`: input paths, SHA-256 checksums, observed dataset properties, exact protocol and scope notes.
- `outputs/*_seed_results.csv`: one row per seed and configuration.
- `outputs/*_summary.csv`: mean, standard deviation, and 95% confidence-interval half-widths.
- `outputs/main_paired_tests.csv`: matched-seed paired differences with Holm--Bonferroni adjustment.

