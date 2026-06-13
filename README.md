# Crop Recommendation System - Togo Agro-climatic Zones

Multi-class machine learning pipeline for smallholder crop advisory in Togo.
Given soil properties, climate variables, and agro-climatic zone, the model recommends the most suitable crop among five staple crops cultivated in the country.

---

## Problem Statement

Smallholder farmers in Togo face yield losses partly due to suboptimal crop selection relative to local soil and climate conditions. This project frames crop recommendation as a supervised multi-class classification problem, using agro-climatic features as predictors.

This work is developed as a methodological prototype in preparation for graduate research on precision agriculture and climate-adaptive farming systems in West Africa (UTFPR application, 2025).

---

## Crops and Agro-climatic Zones

**Target classes:** Maize, Cassava, Sorghum, Yam, Soybean

**Regions modeled:**

| Zone      | Mean Rainfall (mm/yr) | Mean Temp. (C) | Dominant crops        |
|-----------|----------------------|----------------|-----------------------|
| Savane    | 750                  | 29.5           | Maize, Sorghum        |
| Kara      | 1050                 | 27.5           | Maize, Cassava        |
| Centrale  | 1100                 | 27.0           | Maize, Yam, Cassava   |
| Plateaux  | 1300                 | 25.5           | Cassava, Yam          |
| Maritime  | 1450                 | 27.0           | Cassava               |

---

## Features

| Feature         | Unit      | Description                              |
|----------------|-----------|------------------------------------------|
| rainfall_mm     | mm/year   | Annual cumulative rainfall               |
| temp_mean_c     | C         | Mean annual temperature                  |
| soil_ph         | -         | Soil pH (4.5 - 8.0)                      |
| soil_n_ppm      | ppm       | Soil nitrogen content                    |
| soil_p_ppm      | ppm       | Soil phosphorus content                  |
| soil_k_ppm      | ppm       | Soil potassium content                   |
| humidity_pct    | %         | Mean relative humidity                   |
| solar_rad_mj    | MJ/m2/day | Mean daily solar radiation               |
| zone            | category  | Agro-climatic zone (5 classes)           |

---

## Model Architecture

Soft-voting ensemble combining three base learners:

- Random Forest (200 trees, max depth 12)
- Gradient Boosting (150 estimators, learning rate 0.08)
- Logistic Regression (L2 regularization)

Preprocessing: StandardScaler for numeric features, OneHotEncoder for zone.

---

## Results

| Metric                  | Value              |
|------------------------|--------------------|
| CV Accuracy (5-fold)   | 0.881 +/- 0.025    |
| Test Accuracy          | 0.905              |
| Test F1 (weighted)     | 0.905              |

Per-class F1 scores (test set):

| Crop    | Precision | Recall | F1   |
|---------|-----------|--------|------|
| Cassava | 0.95      | 0.95   | 0.95 |
| Maize   | 0.87      | 0.90   | 0.88 |
| Sorghum | 0.96      | 0.86   | 0.91 |
| Soybean | 0.94      | 0.92   | 0.93 |
| Yam     | 0.85      | 0.86   | 0.86 |

---

## Repository Structure

```
crop-recommendation-togo/
├── Crop_Recommendation_Togo.ipynb   # Main notebook (Colab-ready)
├── generate_data.py                 # Synthetic dataset generator
├── train_evaluate.py                # Full pipeline with figures
├── README.md
└── figures/
    ├── 01_eda_agro_features.png     # Feature distributions by crop
    └── 02_results_dashboard.png     # CV, feature importance, confusion matrix
```

---

## How to Run

**Google Colab (recommended):**

1. Go to colab.research.google.com
2. File -> Upload notebook -> select `Crop_Recommendation_Togo.ipynb`
3. Runtime -> Run all

**Local:**

```bash
pip install numpy pandas scikit-learn matplotlib
python generate_data.py
python train_evaluate.py
```

---

## Data Note

The dataset is synthetic, generated from parametric distributions calibrated on published agronomic data for Togo and West Africa (FAO GAEZ, IITA crop guides). A 3% label noise was applied to simulate real-world field annotation variability. The simulation is intended for methodological validation only; production deployment would require ground-truth field survey data.

---

## Academic Context

This prototype is part of a portfolio developed for the UTFPR graduate research application (Precision Agriculture and Remote Sensing, 2025). The target architecture mirrors the feature engineering and ensemble approach described in the research proposal, specifically Phase 2 (multi-crop recommendation from soil-climate inputs).

---

## References

- FAO (2015). GAEZ - Global Agro-Ecological Zones v3.0. Food and Agriculture Organization.
- IITA (2020). Crop management guidelines for West Africa. International Institute of Tropical Agriculture.
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics, 29(5), 1189-1232.
- Mucherino, A., Papajorgji, P., & Pardalos, P. M. (2009). Data Mining in Agriculture. Springer.
