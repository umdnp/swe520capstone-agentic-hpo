from __future__ import annotations

from typing import Any

import pandas as pd

# -------------------------------------------------------------------
# Annotation configuration
# -------------------------------------------------------------------
# For each categorical column, we define:
#   - "categories": the canonical category list
#   - "mapping": mapping from raw values -> canonical
# -------------------------------------------------------------------
ANNOTATION_CONFIG: dict[str, dict[str, Any]] = {
    "admissiondx_category": {
        "categories": ["cardiac", "hepatic", "neurologic", "other", "respiratory", "sepsis", "trauma"],
        "mapping": {
            "cardiac": "cardiac",
            "hepatic": "hepatic",
            "neurologic": "neurologic",
            "other": "other",
            "respiratory": "respiratory",
            "sepsis": "sepsis",
            "trauma": "trauma",
        },
    },
    "age_group": {
        "categories": ["elderly", "middle", "older", "young"],
        "mapping": {
            "elderly": "elderly",
            "middle": "middle",
            "older": "older",
            "young": "young",
        },
    },
    "ethnicity": {
        "categories": ["African American", "Asian", "Caucasian", "Hispanic", "Native American", "Other/Unknown"],
        "mapping": {
            "african american": "African American",
            "asian": "Asian",
            "caucasian": "Caucasian",
            "hispanic": "Hispanic",
            "native american": "Native American",
            "other/unknown": "Other/Unknown",
        },
    },
    "gender": {
        "categories": ["female", "male", "other", "unknown"],
        "mapping": {
            "f": "female",
            "female": "female",
            "m": "male",
            "male": "male",
            "nb": "other",
            "non-binary": "other",
            "other": "other",
            "unknown": "unknown",
        },
    },
    "hospital_region": {
        "categories": ["Midwest", "Northeast", "South", "West"],
        "mapping": {
            "midwest": "Midwest",
            "northeast": "Northeast",
            "south": "South",
            "west": "West",
        },
    },
    "hospitaladmitsource": {
        "categories": ["Acute Care/Floor", "Chest Pain Center", "Direct Admit", "Emergency Department", "Floor",
                       "ICU", "ICU to SDU", "Observation", "Operating Room", "Other Hospital", "Other ICU", "PACU",
                       "Recovery Room", "Step-Down Unit (SDU)", "Other"],
        "mapping": {
            "acute care/floor": "Acute Care/Floor",
            "chest pain center": "Chest Pain Center",
            "direct admit": "Direct Admit",
            "emergency department": "Emergency Department",
            "floor": "Floor",
            "icu": "ICU",
            "icu to sdu": "ICU to SDU",
            "observation": "Observation",
            "operating room": "Operating Room",
            "other hospital": "Other Hospital",
            "other icu": "Other ICU",
            "pacu": "PACU",
            "recovery room": "Recovery Room",
            "step-down unit (sdu)": "Step-Down Unit (SDU)",
            "other": "Other",
        },
    },
    "numbedscategory": {
        "categories": ["low", "lowmid", "highmid", "high"],
        "mapping": {
            "<100": "low",
            "100 - 249": "lowmid",
            "250 - 499": "highmid",
            ">= 500": "high",
        },
    },
    "teachingstatus": {
        "categories": ["1", "0"],
        "mapping": {
            "true": "1",
            "false": "0",
        },
    },
    "unitadmitsource": {
        "categories": ["Acute Care/Floor", "Chest Pain Center", "Direct Admit", "Emergency Department", "Floor",
                       "ICU", "ICU to SDU", "Observation", "Operating Room", "Other Hospital", "Other ICU", "PACU",
                       "Recovery Room", "Step-Down Unit (SDU)", "Other"],
        "mapping": {
            "acute care/floor": "Acute Care/Floor",
            "chest pain center": "Chest Pain Center",
            "direct admit": "Direct Admit",
            "emergency department": "Emergency Department",
            "floor": "Floor",
            "icu": "ICU",
            "icu to sdu": "ICU to SDU",
            "observation": "Observation",
            "operating room": "Operating Room",
            "other hospital": "Other Hospital",
            "other icu": "Other ICU",
            "pacu": "PACU",
            "recovery room": "Recovery Room",
            "step-down unit (sdu)": "Step-Down Unit (SDU)",
            "other": "Other",
        },
    },
    "unittype": {
        "categories": ["Cardiac ICU", "CCU-CTICU", "CTICU", "CSICU", "Med-Surg ICU", "MICU", "Neuro ICU", "SICU"],
        "mapping": {
            "cardiac icu": "Cardiac ICU",
            "ccu-cticu": "CCU-CTICU",
            "cticu": "CTICU",
            "csicu": "CSICU",
            "med-surg icu": "Med-Surg ICU",
            "micu": "MICU",
            "neuro icu": "Neuro ICU",
            "sicu": "SICU",
        },
    },
}


def _normalize_raw_value(val: Any) -> str | None:
    """
    Normalize raw categorical values to lowercase strings for lookup.
    """
    if val is None:
        return None

    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        return s.lower()

    # for non-strings, convert to string then lowercase
    return str(val).strip().lower() or None


def _choose_fallback(categories: list[str]) -> str:
    """
    Pick a fallback label that matches one of the configured categories.
    Preference: unknown -> other -> last category.
    """
    for candidate in ("unknown", "Unknown", "other", "Other"):
        if candidate in categories:
            return candidate
    return categories[-1] if categories else "unknown"


def annotate_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply annotation rules to known categorical columns.
    """
    df = df.copy()

    for col, cfg in ANNOTATION_CONFIG.items():
        if col not in df.columns:
            continue  # skip if column not found in dataset

        categories = cfg["categories"]
        mapping = cfg["mapping"]

        # determine fallback category
        fallback = _choose_fallback(categories)

        # normalize raw values
        normalized = df[col].map(_normalize_raw_value)

        def map_value(raw_norm: str | None) -> str:
            if raw_norm is None:
                return fallback
            return mapping.get(raw_norm, fallback)

        df[col] = normalized.map(map_value).astype("category")

        invalid = set(df[col].dropna().unique()) - set(categories)
        if invalid:
            raise RuntimeError(f"{col} produced values not in categories: {sorted(invalid)}")

    return df
