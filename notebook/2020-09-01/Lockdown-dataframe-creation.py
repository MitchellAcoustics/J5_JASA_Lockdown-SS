"""Onload and separate out the required data for the Lockdown model building
"""

import os
import sys

sys.path.append(
    "C:\\Users\\Andrew\\OneDrive - University College London\\_PhD\\Papers - Drafts\\J5_JASA_Lockdown-SS"
)

from scripts import lockdown_mlm as mlm
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


# Define some constants and options
## variables
dep_vars = [
    "Natural",
    "Traffic",
    "Human",
    "Other",
    "loudness",
    "overall",
    "Pleasant",
    "Eventful",
]

FEATS_LISTS = mlm.FEATS_LISTS
# remove = ["FS_TEMP", "LAeq_TEMP", "LCeq_TEMP", "LZeq_TEMP", "I_TEMP", "N_TEMP", "R_TEMP", "S_TEMP", "SIL_TEMP", "THD_TEMP", "T_TEMP"]

# for k in remove:
#     FEATS_LISTS.pop(k, None)

acoustic_vars = sorted({x for v in FEATS_LISTS.values() for x in v})

## processing options
nonlinear_transformations = []  # Leave a blank list to do no transformations

# ##################################################################
# Load Data

DATA_DIR = Path(
    "C:/Users/Andrew/OneDrive - University College London/_PhD/Papers - Drafts/J5_JASA_Lockdown-SS/data"
)
# RESULTS_DIR = Path("C:/Users/Andrew/OneDrive - University College London/_PhD/Papers - Drafts/J5_JASA_Lockdown-SS/results")
ssidData = pd.read_csv(
    DATA_DIR.joinpath("2020-08-13/LondonVeniceBINResults_2020-08-13_4.csv")
)

for col_name in ["Lockdown"]:
    ssidData[col_name] = ssidData[col_name].astype("category")

# Cutdown the dataset
cols = ["GroupID", "LocationID", "SessionID", "Lockdown"] + dep_vars + acoustic_vars
ssidData = ssidData[cols]

# Compress to mean of each GroupID
# compressData = ssidData.copy()
compressData = ssidData.groupby(["GroupID"]).mean()
compressData = compressData.merge(
    ssidData[["GroupID", "LocationID", "SessionID", "Lockdown"]].drop_duplicates(),
    on="GroupID",
)

location_codes = pd.Categorical(compressData["LocationID"]).codes
compressData["LocationID_codes"] = location_codes
compressData.loc[compressData["Lockdown"] == 1].dropna(inplace=True)
compressData = compressData.dropna(subset=acoustic_vars)

compressData, acoustic_vars = mlm.nonlinear_features(
    compressData, acoustic_vars, transformations=nonlinear_transformations
)
print("Initial features number:    ", len(acoustic_vars))

# save unscaled dataframe
compressData.to_csv(DATA_DIR.joinpath("2020-09-01/Lockdown-unscaled-compressData.csv"))

# Standardise
from sklearn.preprocessing import StandardScaler

compressData = compressData.replace([np.inf, -np.inf], np.nan)
compressData = compressData.dropna(subset=acoustic_vars)
scaler = StandardScaler()
compressData[acoustic_vars] = scaler.fit_transform(compressData[acoustic_vars])

compressData.to_csv(DATA_DIR.joinpath("2020-09-01/Lockdown-scaled-compressData.csv"))

# ###############################################################
# Split Prelockdown from during lockdown
prelockdownData = compressData.loc[compressData["Lockdown"] == 1]
prelockdownData = prelockdownData.dropna()
print("prelockdownData shape:     ", prelockdownData.shape)
lockdownData = compressData.loc[compressData["Lockdown"] == 2]
print("during lockdownData shape: ", lockdownData.shape)

prelockdownData.to_csv(DATA_DIR.joinpath("2020-09-01/preLockdownData.csv"))
lockdownData.to_csv(DATA_DIR.joinpath("2020-09-01/lockdownData.csv"))

# Split into training and testing sets
prelockdownTrain, prelockdownTest = train_test_split(
    prelockdownData, test_size=0.25, random_state=42
)

prelockdownTrain.to_csv(DATA_DIR.joinpath("2020-09-01/prelockdownDataTrain.csv"))
prelockdownTest.to_csv(DATA_DIR.joinpath("2020-09-01/prelockdownDataTest.csv"))
