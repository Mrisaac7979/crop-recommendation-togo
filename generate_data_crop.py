"""
generate_data.py
Synthetic agronomic dataset for crop recommendation in Togo.
Features reflect agro-climatic zones: Savane, Kara, Centrale, Plateaux, Maritime.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

N = 2000

CROPS = ["Maize", "Cassava", "Sorghum", "Yam", "Soybean"]

ZONE_PARAMS = {
    "Savane": {
        "rainfall_mm":    (750,  120),
        "temp_mean_c":    (29.5, 1.5),
        "soil_ph":        (6.2,  0.5),
        "soil_n_ppm":     (120,  30),
        "soil_p_ppm":     (22,   8),
        "soil_k_ppm":     (180,  40),
        "humidity_pct":   (52,   8),
        "solar_rad":      (21,   2),
        "crop_prior":     [0.35, 0.20, 0.25, 0.10, 0.10],
    },
    "Kara": {
        "rainfall_mm":    (1050, 130),
        "temp_mean_c":    (27.5, 1.8),
        "soil_ph":        (6.5,  0.4),
        "soil_n_ppm":     (145,  35),
        "soil_p_ppm":     (28,   9),
        "soil_k_ppm":     (210,  45),
        "humidity_pct":   (63,   9),
        "solar_rad":      (19,   2),
        "crop_prior":     [0.30, 0.25, 0.15, 0.15, 0.15],
    },
    "Centrale": {
        "rainfall_mm":    (1100, 140),
        "temp_mean_c":    (27.0, 1.6),
        "soil_ph":        (6.6,  0.4),
        "soil_n_ppm":     (155,  35),
        "soil_p_ppm":     (30,   10),
        "soil_k_ppm":     (220,  50),
        "humidity_pct":   (67,   8),
        "solar_rad":      (18,   2),
        "crop_prior":     [0.30, 0.30, 0.10, 0.20, 0.10],
    },
    "Plateaux": {
        "rainfall_mm":    (1300, 150),
        "temp_mean_c":    (25.5, 1.5),
        "soil_ph":        (6.8,  0.3),
        "soil_n_ppm":     (170,  40),
        "soil_p_ppm":     (35,   10),
        "soil_k_ppm":     (240,  50),
        "humidity_pct":   (74,   7),
        "solar_rad":      (17,   2),
        "crop_prior":     [0.25, 0.35, 0.05, 0.25, 0.10],
    },
    "Maritime": {
        "rainfall_mm":    (1450, 160),
        "temp_mean_c":    (27.0, 1.4),
        "soil_ph":        (5.9,  0.5),
        "soil_n_ppm":     (160,  38),
        "soil_p_ppm":     (32,   10),
        "soil_k_ppm":     (230,  55),
        "humidity_pct":   (80,   6),
        "solar_rad":      (17,   2),
        "crop_prior":     [0.20, 0.40, 0.05, 0.20, 0.15],
    },
}

ZONES = list(ZONE_PARAMS.keys())
zone_counts = np.random.multinomial(N, [1/5]*5)

rows = []
for zone, count in zip(ZONES, zone_counts):
    p = ZONE_PARAMS[zone]
    for _ in range(count):
        crop = np.random.choice(CROPS, p=p["crop_prior"])

        # Inject crop-specific signal to make classification learnable
        crop_offset = {
            "Maize":   {"rainfall_mm": 250,  "soil_n_ppm": 100, "soil_p_ppm": 25,  "temp_mean_c": 2.0,  "soil_ph":  0.5},
            "Cassava": {"rainfall_mm": -200, "soil_n_ppm": -70, "soil_p_ppm": -18, "temp_mean_c": 1.5,  "soil_ph": -0.6},
            "Sorghum": {"rainfall_mm": -380, "soil_n_ppm": -90, "soil_p_ppm": -20, "temp_mean_c": 4.0,  "soil_ph":  0.2},
            "Yam":     {"rainfall_mm": 320,  "soil_n_ppm": 110, "soil_p_ppm": 32,  "temp_mean_c": -2.0, "soil_ph": -0.3},
            "Soybean": {"rainfall_mm": 150,  "soil_n_ppm": 140, "soil_p_ppm": 45,  "temp_mean_c": -3.5, "soil_ph":  0.7},
        }
        off = crop_offset[crop]

        row = {
            "zone":           zone,
            "rainfall_mm":    max(300, np.random.normal(p["rainfall_mm"][0] + off["rainfall_mm"], p["rainfall_mm"][1])),
            "temp_mean_c":    np.random.normal(p["temp_mean_c"][0] + off["temp_mean_c"], p["temp_mean_c"][1]),
            "soil_ph":        np.clip(np.random.normal(p["soil_ph"][0] + off["soil_ph"], p["soil_ph"][1]), 4.5, 8.0),
            "soil_n_ppm":     max(50, np.random.normal(p["soil_n_ppm"][0] + off["soil_n_ppm"], p["soil_n_ppm"][1])),
            "soil_p_ppm":     max(5,  np.random.normal(p["soil_p_ppm"][0] + off["soil_p_ppm"], p["soil_p_ppm"][1])),
            "soil_k_ppm":     max(80, np.random.normal(p["soil_k_ppm"][0], p["soil_k_ppm"][1])),
            "humidity_pct":   np.clip(np.random.normal(p["humidity_pct"][0], p["humidity_pct"][1]), 30, 99),
            "solar_rad_mj":   max(10, np.random.normal(p["solar_rad"][0], p["solar_rad"][1])),
            "label_noise":    np.random.rand(),
            "crop":           crop,
        }
        rows.append(row)

df = pd.DataFrame(rows)

# 5% label noise for realism
flip_idx = df[df["label_noise"] < 0.03].index
df.loc[flip_idx, "crop"] = np.random.choice(CROPS, size=len(flip_idx))
df.drop(columns=["label_noise"], inplace=True)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["crop"])

train_df.to_csv("train_crop.csv", index=False)
test_df.to_csv("test_crop.csv", index=False)

print(f"Train: {len(train_df)} samples | Test: {len(test_df)} samples")
print("Class distribution (train):")
print(train_df["crop"].value_counts())
