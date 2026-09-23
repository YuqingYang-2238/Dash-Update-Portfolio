# Joint Classification and Redshift Estimation of Supernova Spectra with Transformers

**➡️ [View the full project portfolio (live site)](https://yuqingyang-2238.github.io/Dash-Update-Portfolio/)**

A transformer encoder framework that classifies supernova spectra across five major types (Ia, II, IIn, Ib/c, SLSNe-I), trained and evaluated on real WISeREP spectra. It reads spectra on their native wavelength grid — no de-redshifting, continuum removal, or resampling — and outperforms existing methods such as DASH (Muthukrishna et al., 2019), lifting rare-class IIn recall from 0.09 to 0.70 with 88% overall validation accuracy. Includes four variants: a baseline transformer, a simulation-pretrained transformer, and two redshift-aware models for when the true redshift is missing.

Accepted as a poster at **Open SkAI 2025**.

## Author

**Yuqing Yang** — PhD student, Statistics & Data Science, Northwestern University
yuqingyang2029@u.northwestern.edu · [@YuqingYang-2238](https://github.com/YuqingYang-2238)

*Released under the MIT License (see `LICENSE`).*
