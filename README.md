# Joint Classification and Redshift Estimation of Supernova Spectra with Transformers


![Conference](https://img.shields.io/badge/Open%20SkAI%202025-Poster-1565C0)
![Framework](https://img.shields.io/badge/PyTorch-Transformer-2E7D32)
![Task](https://img.shields.io/badge/Tasks-Spectra%20Classification%20%2B%20Redshift%20Estimation-00838F)
![License](https://img.shields.io/badge/License-MIT-555555)

**TL;DR** — Massive fiber-fed spectroscopic surveys such as 4MOST will obtain hundreds of bright supernovae and other transient events, following discoveries by the Vera C. Rubin Observatory. Classifying these spectra probabilistically is crucial to incorporating them into population studies and cosmological analyses. We are constructing a novel transformer encoder-based framework for classifying these transient spectra and estimating their redshift. Our model supports variable-length spectra through masking and encodes wavelength information using sinusoidal positional embeddings, enabling robust performance on spectra with irregularly spaced wavelength coverage—an inherent challenge in real observational data. By evaluating on real supernova spectra from the Weizmann Interactive Supernova Data Repository (WISeREP) database, our approach outperforms existing methods (e.g. DASH - Muthukrishna et al., 2019) across key metrics. 
This work was accepted as a poster at Open SkAI 2025, hosted by NSF-Simons AI Institute for the Sky (SkAI), 
funded by the U.S. National Science Foundation and the Simons Foundation.

## The problem

When a massive star dies, it explodes as a **supernova**. The light from that explosion, spread out into a **spectrum** (brightness as a function of wavelength), carries the chemical and physical fingerprints of the event. Astronomers sort supernovae into a handful of physical types: Ia, Ib/c, II, IIn, and the superluminous SLSNe‑I, and each type tells a different story about how the star died. Type Ia supernovae are the "standard candles" used to measure the expansion of the Universe, and the spectrum also encodes **redshift**, which tells us how far away the explosion happened and how much the Universe has stretched since its light set out.
The catch is volume. Surveys such as the Vera C. Rubin Observatory's LSST will discover **millions** of transient events — far more than anyone could ever classify by hand, and only a small fraction can be followed up spectroscopically. Fast, reliable, automated classification of spectra, including the rare, faint, and scientifically precious types, is a genuine bottleneck for modern astronomy.


![Classification taxonomy](figures/taxonomy.png)

## Why existing methods fall short

Template‑matching tools (such as SNID) and the popular deep‑learning classifier **DASH** (Muthukrishna et al., 2019) demand heavily pre‑processed input: de‑redshifting, continuum removal, and resampling every spectrum onto a fixed, evenly‑spaced wavelength grid. I began this project by benchmarking DASH from scratch and documenting concrete failure modes:

- it **overfits the common classes and collapses on the rare ones** — recall on the IIn class drops to roughly **0.09**;
- its accuracy is **sensitive to redshift**, degrading on higher‑redshift spectra;
- it **cannot accept variable wavelength ranges or spacings**, so it can't generalise across instruments;
- it tends to **misclassify noisy spectra into the Ic group**.

## My approach

I reframed the problem so a **single transformer** handles both tasks: transient classification and redshift estimation. A spectrum is treated as a sequence: the **flux** values are the tokens and the **wavelength** is encoded as position, so the model reads spectra on their native grid without de‑redshifting or resampling. A redshift value can be supplied as metadata when it is available.

![Model architecture](figures/architecture.png)

The **backbone** is a 6‑layer transformer encoder (model dimension 128, 8 attention heads) followed by a **Perceiver‑style cross‑attention bottleneck** that compresses a variable‑length spectrum into a compact latent representation. Lightweight heads sit on top: an MLP **classifier** and/or a **redshift regressor** — either an MLP, or a **Mixture Density Network** when calibrated uncertainty is wanted. Class imbalance is handled with **focal loss**.

I explored four variants to learn the best way to combine scarce real data with abundant simulations:

| Variant | Idea | Predicts |
|---|---|---|
| **Baseline** | Train directly on real Wiserep spectra, true redshift as metadata | Type |
| **Transfer Learning** | Pretrain the backbone on balanced SASSAFRAS simulations, then fine‑tune on real data | Type |
| **Model_Z** | Separate redshift and classification heads; the classifier can consume an *estimated* redshift, so it still works when the true value is missing | Type + redshift |
| **Model_mt** | A single multi‑task head that predicts type and redshift **jointly** | Type + redshift |

Training data is ~4,800 real spectra plus class‑balanced simulations, augmented by oversampling each rare class to ~1,000 examples, adding white noise, random wavelength cropping, and random redshifting.

## Results

The headline is **per‑class recall**. Across every transformer variant, the largest gains land precisely on the classes DASH handles worst.

![Per-class recall versus DASH](figures/recall_by_class.png)

The most dramatic improvement is on the rare **IIn** class, where recall climbs from **0.09 (DASH) to 0.70 (Baseline)** — a class DASH effectively cannot recover. The transformer variants also hold strong recall on Ia (≥ 0.91) and improve the commonly‑confused II class, all **without** the de‑redshifting and fixed‑grid preprocessing that DASH requires.

![Baseline confusion matrix](figures/confusion_matrix.png)

The **Baseline** model reaches **88% overall validation accuracy** with a clean diagonal across all five types. **Transfer Learning** matches that 88% and edges ahead on the II class (0.85 recall), while **Model_mt** (0.85) and **Model_Z** (0.81) additionally output a redshift estimate — the multi‑task model tracks the true redshift closely across nearly two decades in *z*. All four variants are compared side by side below.

![Confusion matrices for all four variants](figures/confusion_grid.png)

## My contributions

I designed and built this project end to end, and I am the **sole author and maintainer** of the codebase:

- **Designed the training pipeline** — data loaders for heterogeneous spectra, augmentation and minority oversampling, focal‑loss imbalance handling, and a fully config‑driven training/evaluation framework.
- **Implemented the transformer architecture** — the encoder backbone with a Perceiver‑style cross‑attention bottleneck, plus the classification and redshift heads (MLP and Mixture Density Network) in PyTorch.
- **Developed the four modelling strategies** — the baseline, the simulation‑pretraining transfer‑learning model, the separate‑heads Model_Z, and the joint multi‑task model.
- **Conducted the experiments** — including the from‑scratch benchmarking and failure analysis of DASH that motivated the whole redesign.
- **Analysed the results** — per‑class recall, confusion matrices, and redshift‑binned evaluation.

## Conference

An abstract based on this work was **accepted as a poster at the inaugural Open SkAI 2025 conference**, held at the SkAI Hub in Chicago, Illinois, USA, on September 2–5, 2025.

## Author

**Yuqing Yang** — Northwestern University, Department of Statistics & Data Science
GitHub: [@YuqingYang-2238](https://github.com/YuqingYang-2238)

*Released under the MIT License (see `LICENSE`).*
