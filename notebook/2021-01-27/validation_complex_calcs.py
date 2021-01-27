#%% Load in data
from pathlib import Path
import pandas as pd
import numpy as np

CATEGORISED_VARS = {
    "indexing": [
        "GroupID",
        "SessionID",
        "LocationID",
        "Country",
        "record_id",
    ],  # Ways to index which survey it is
    "meta_info": [
        "recording",
        "start_time",
        "end_time",
        "longitude",
        "latitude",
    ],  # Info from when that survey was collected
    "sound_source_dominance": [
        "Traffic",
        "Other",
        "Human",
        "Natural",
    ],  # Sound sources
    "complex_PAQs": ["Pleasant", "Eventful"],  # Projected PAQ coordinates
    "raw_PAQs": [
        "pleasant",
        "chaotic",
        "vibrant",
        "uneventful",
        "calm",
        "annoying",
        "eventful",
        "monotonous",
    ],  # Raw 5-point PAQs
    "overall_soundscape": [
        "overall",
        "appropriateness",
        "loudness",
        "often",
        "visit_again",
    ],  # Questions about the overall soundscape
    "demographics": ["Age", "Gender", "Occupation", "Education", "Ethnicity", "Resid"],
    "misc": ["AnythingElse"],
}


# Survey database functions
def fill_missing_paqs(df, features, fill_val=3):
    df[features] = df[features].fillna(value=fill_val)
    return df


def calculate_complex_paqs(
    df, scale_to_one: bool = True, fill_na: bool = True, fill_val=3, append_var_names=""
):
    """Calculate the complex Pleasant and Eventful projections of the PAQs.
    Uses the projection formulae from ISO  12913 Part 3:

    P =(p−a)+cos45°*(ca−ch)+cos45°*(v−m)
    E =(e−u)+cos45°*(ch−ca)+cos45°*(v−m)

    Parameters
    ----------
    scale_to_one : bool, optional
        Scale the complex values from -1 to 1, by default True
    fill_na : bool, optional
        Fill missing raw_PAQ values, by default True
    fill_val : int, optional
        Value to fill missing raw_PAQs with, by default 3

    Returns
    -------
    (pd.Series, pd.Series)
        pandas Series containing the new complex Pleasant and Eventful vectors
    """
    features = CATEGORISED_VARS["raw_PAQs"]
    features = [var + append_var_names for var in features]

    if fill_na:
        df = fill_missing_paqs(df, features, fill_val=fill_val)

    # TODO: Add check for raw_PAQ column names
    # TODO: add handling for if sf already contains Pleasant and Eventful values

    proj = np.cos(np.deg2rad(45))
    scale = 4 + np.sqrt(32)

    # TODO: Add if statements for too much missing data
    # P =(p−a)+cos45°(ca−ch)+cos45°(v−m)
    complex_pleasant = (
        (df["pleasant" + append_var_names] - df["annoying" + append_var_names])
        + proj * (df["calm" + append_var_names] - df["chaotic" + append_var_names])
        + proj
        * (df["vibrant" + append_var_names] - df["monotonous" + append_var_names])
    )
    Pleasant = complex_pleasant / scale if scale_to_one else complex_pleasant

    # E =(e−u)+cos45°(ch−ca)+cos45°(v−m)
    complex_eventful = (
        (df["eventful" + append_var_names] - df["uneventful" + append_var_names])
        + proj * (df["chaotic" + append_var_names] - df["calm" + append_var_names])
        + proj
        * (df["vibrant" + append_var_names] - df["monotonous" + append_var_names])
    )
    Eventful = complex_eventful / scale if scale_to_one else complex_eventful

    return Pleasant, Eventful


def test_calculate_complex_paqs():
    test_df = pd.DataFrame(
        columns=[
            "pleasant",
            "vibrant",
            "eventful",
            "chaotic",
            "annoying",
            "monotonous",
            "uneventful",
            "calm",
        ],
        index=range(1),
    )
    test_df[
        [
            "pleasant",
            "vibrant",
            "eventful",
            "chaotic",
            "annoying",
            "monotonous",
            "uneventful",
            "calm",
        ]
    ] = [5, 5, 5, 5, 1, 1, 1, 1]
    Pleasant, Eventful = calculate_complex_paqs(
        test_df, scale_to_one=True, fill_na=False
    )
    assert Eventful.values == 1

    test_df = pd.DataFrame(
        columns=[
            "pleasant",
            "vibrant",
            "eventful",
            "chaotic",
            "annoying",
            "monotonous",
            "uneventful",
            "calm",
        ],
        index=range(1),
    )
    test_df[
        [
            "pleasant",
            "vibrant",
            "eventful",
            "chaotic",
            "annoying",
            "monotonous",
            "uneventful",
            "calm",
        ]
    ] = [5, 5, 1, 1, 1, 1, 1, 5]
    Pleasant, Eventful = calculate_complex_paqs(
        test_df, scale_to_one=True, fill_na=False
    )
    assert Pleasant.values == 1

    test_df = pd.DataFrame(
        columns=[
            "pleasant_a",
            "vibrant_a",
            "eventful_a",
            "chaotic_a",
            "annoying_a",
            "monotonous_a",
            "uneventful_a",
            "calm_a",
        ],
        index=range(1),
    )
    test_df[
        [
            "pleasant_a",
            "vibrant_a",
            "eventful_a",
            "chaotic_a",
            "annoying_a",
            "monotonous_a",
            "uneventful_a",
            "calm_a",
        ]
    ] = [5, 5, 1, 1, 1, 1, 1, 5]
    Pleasant, Eventful = calculate_complex_paqs(
        test_df, scale_to_one=True, fill_na=False, append_var_names="_a"
    )
    assert Pleasant.values == 1


#%%

datafile = Path("../../data/2021-01-27/validation_wooutliers.xlsx")

df = pd.read_excel(datafile, header=0)
df.head()

#%%

test_calculate_complex_paqs()

Pleasant, Eventful = calculate_complex_paqs(df, append_var_names="_A")
df["Pleasant"] = Pleasant
df["Eventful"] = Eventful
df.head()

# %% save file

from datetime import date

df.to_excel(
    f"validation_wooutliers_Complex_calcs_{date.today().isoformat()}.xlsx",
    index=False,
    sheet_name="validation_wooutliers",
)

# %%
