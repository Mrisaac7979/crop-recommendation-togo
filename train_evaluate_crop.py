"""
train_evaluate.py
Multi-class crop recommendation model for Togo agro-climatic zones.
Pipeline: preprocessing -> ensemble (RF + GB + LR) -> evaluation.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
os.makedirs("figures", exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
train_df = pd.read_csv("train_crop.csv")
test_df  = pd.read_csv("test_crop.csv")

NUMERIC_FEATURES = [
    "rainfall_mm", "temp_mean_c", "soil_ph",
    "soil_n_ppm", "soil_p_ppm", "soil_k_ppm",
    "humidity_pct", "solar_rad_mj"
]
CATEGORICAL_FEATURES = ["zone"]
TARGET = "crop"

X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_train = train_df[TARGET]
X_test  = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_test  = test_df[TARGET]

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
CLASSES = le.classes_

# ── 2. Preprocessor ──────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
])

# ── 3. Ensemble ───────────────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=SEED, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=SEED)
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
    voting="soft"
)

pipeline = Pipeline([("prep", preprocessor), ("model", ensemble)])

# ── 4. Cross-validation ───────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(pipeline, X_train, y_train_enc, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ── 5. Final fit & test evaluation ───────────────────────────────────────────
pipeline.fit(X_train, y_train_enc)
y_pred = pipeline.predict(X_test)

acc  = accuracy_score(y_test_enc, y_pred)
f1   = f1_score(y_test_enc, y_pred, average="weighted")
print(f"Test Accuracy : {acc:.4f}")
print(f"Test F1 (weighted): {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_enc, y_pred, target_names=CLASSES))

# ── 6. Feature importance (RF component) ──────────────────────────────────────
rf_fitted = pipeline.named_steps["model"].estimators_[0]
ohe_cols  = pipeline.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
feature_names = NUMERIC_FEATURES + list(ohe_cols)
importances = rf_fitted.feature_importances_

# ── 7. Figure 1 — EDA ─────────────────────────────────────────────────────────
COLORS = {
    "Maize":   "#e6a817",
    "Cassava": "#2e8b57",
    "Sorghum": "#c0392b",
    "Yam":     "#8e44ad",
    "Soybean": "#2980b9",
}

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle("Exploratory Data Analysis - Agro-climatic Features by Crop", fontsize=13, y=1.01)

plot_features = [
    ("rainfall_mm",   "Rainfall (mm)"),
    ("temp_mean_c",   "Mean Temperature (C)"),
    ("soil_ph",       "Soil pH"),
    ("soil_n_ppm",    "Soil Nitrogen (ppm)"),
    ("soil_p_ppm",    "Soil Phosphorus (ppm)"),
    ("soil_k_ppm",    "Soil Potassium (ppm)"),
    ("humidity_pct",  "Humidity (%)"),
    ("solar_rad_mj",  "Solar Radiation (MJ/m2)"),
]

for idx, (feat, label) in enumerate(plot_features):
    ax = axes[idx // 3][idx % 3]
    for crop in CLASSES:
        vals = train_df[train_df[TARGET] == crop][feat]
        ax.hist(vals, bins=25, alpha=0.55, color=COLORS[crop], label=crop)
    ax.set_title(label, fontsize=9)
    ax.tick_params(labelsize=7)

# Zone distribution in last cell
ax = axes[2][2]
zone_crop = train_df.groupby(["zone", TARGET]).size().unstack(fill_value=0)
zone_crop.plot(kind="bar", ax=ax, color=list(COLORS.values()), legend=False, width=0.75)
ax.set_title("Crop distribution by zone", fontsize=9)
ax.set_xlabel("")
ax.tick_params(axis="x", labelsize=6, rotation=30)
ax.tick_params(axis="y", labelsize=7)

handles = [plt.Rectangle((0,0),1,1, color=COLORS[c]) for c in CLASSES]
fig.legend(handles, CLASSES, loc="lower center", ncol=5, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, -0.03))

plt.tight_layout()
plt.savefig("figures/01_eda_agro_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 1 saved.")

# ── 8. Figure 2 — Results dashboard ──────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

# (a) CV accuracy per fold
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(range(1, 6), cv_scores, color="#2e86c1", edgecolor="white", linewidth=0.6)
ax1.axhline(cv_scores.mean(), color="#c0392b", linestyle="--", linewidth=1.2, label=f"Mean {cv_scores.mean():.3f}")
ax1.set_title("5-Fold Cross-Validation Accuracy", fontsize=9)
ax1.set_xlabel("Fold", fontsize=8)
ax1.set_ylabel("Accuracy", fontsize=8)
ax1.set_ylim(0.80, 1.0)
ax1.legend(fontsize=7)
ax1.tick_params(labelsize=7)

# (b) Feature importance (top 8)
ax2 = fig.add_subplot(gs[0, 1:])
top8_idx  = np.argsort(importances)[-8:]
top8_names = [feature_names[i].replace("zone_", "zone:") for i in top8_idx]
top8_vals  = importances[top8_idx]
ax2.barh(top8_names, top8_vals, color="#1a5276", edgecolor="white", linewidth=0.5)
ax2.set_title("Top 8 Feature Importances (Random Forest component)", fontsize=9)
ax2.set_xlabel("Mean Decrease in Impurity", fontsize=8)
ax2.tick_params(labelsize=7)

# (c) Confusion matrix
ax3 = fig.add_subplot(gs[1, 0:2])
cm  = confusion_matrix(y_test_enc, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(ax=ax3, colorbar=False, cmap="Blues")
ax3.set_title("Confusion Matrix (test set)", fontsize=9)
ax3.tick_params(labelsize=7)
for label in ax3.get_xticklabels():
    label.set_rotation(25)

# (d) Per-class F1
ax4 = fig.add_subplot(gs[1, 2])
per_class_f1 = f1_score(y_test_enc, y_pred, average=None)
bars = ax4.bar(CLASSES, per_class_f1, color=[COLORS[c] for c in CLASSES], edgecolor="white", linewidth=0.6)
ax4.set_title("Per-class F1 Score (test set)", fontsize=9)
ax4.set_ylabel("F1 Score", fontsize=8)
ax4.set_ylim(0.70, 1.0)
ax4.tick_params(axis="x", labelsize=7, rotation=20)
ax4.tick_params(axis="y", labelsize=7)
for bar, val in zip(bars, per_class_f1):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.2f}", ha="center", va="bottom", fontsize=7)

# Summary text
metrics_text = (
    f"Summary Metrics\n"
    f"CV Accuracy : {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}\n"
    f"Test Accuracy: {acc:.3f}\n"
    f"Test F1 (weighted): {f1:.3f}\n"
    f"N train: {len(train_df)} | N test: {len(test_df)}\n"
    f"Classes: {', '.join(CLASSES)}"
)
fig.text(0.01, 0.98, metrics_text, va="top", ha="left", fontsize=8,
         family="monospace", bbox=dict(boxstyle="round,pad=0.4", fc="#eaf2f8", ec="#2e86c1", lw=0.8))

fig.suptitle("Crop Recommendation Model - Results Dashboard", fontsize=13, y=1.01)
plt.savefig("figures/02_results_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 2 saved.")
print("\nAll done.")
