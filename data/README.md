# Data

The model is trained and evaluated on supernova spectra from two sources:

- **Real spectra — Wiserep / SNID.** ~4,800 observed supernova spectra (rest frame, with phase information), labelled with one of the five target types.
- **Simulated spectra — SASSAFRAS.** Class‑balanced simulations used to pretrain the transformer backbone before fine‑tuning on real data.

## Format

All processed datasets are stored as `.npz` archives with consistent variable names:

```python
import numpy as np
dt        = np.load(path)
wavelength = dt["wavelengths"]   # wavelength grid (used as positional input)
flux       = dt["fluxes"]        # flux values (model input / tokens)
masks      = dt["masks"]         # valid-bin masks for variable-length spectra
redshift   = dt["redshifts"]     # redshift (metadata or regression target)
labels     = dt["labels"]        # class label (Ia, Ib/c, II, IIn, SLSNe-I)
```

Typical processed files: `augmented_trainDataset_standardized.npz`,
`testDataset_wiserep_standardized.npz` (real), and
`trainDataset_SASSAFRAS_standardized_balanced.npz`,
`testDataset_SASSAFRAS_standardized.npz` (simulated).

## Access

- **Processed data** (ready to train/validate): [Google Drive](https://drive.google.com/drive/folders/1kjsHkQVZ1SOoMGscvZc0JDy_e4d1h71C?usp=sharing)
- **Raw data:** available on request from the dataset maintainers (Amanda and Jennifer).

> Data files are not committed to this repository — see `.gitignore`.
