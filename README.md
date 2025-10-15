# Chicago Crime Prediction (2023–present)

This project builds machine-learning models on **live crime data from the City of Chicago** to predict:
1) the **arrest/domestic status** of an incident, and  
2) the **primary crime type**,  
using only coarse context features (location block, month, hour-of-day range, season).

The code fetches ~17k recent records from the public API, performs preprocessing/EDA, trains several models (**Random Forest, Decision Tree, XGBoost, KNN**), evaluates them (Accuracy & weighted **F1**), visualizes a confusion matrix, and generates a **predictions.csv** covering all combinations of location × hour range × month × season.

---

## Dataset & Source

- **API**: `https://data.cityofchicago.org/resource/crimes.json`
- **Query**: incidents with `year >= 2023`, capped with `$limit=17000`
- **Fetched columns (normalized)**: `primary_type`, `block`, `arrest`, `domestic`, `date`, …
- **License/usage**: follow Chicago open data terms.

> Data is pulled at runtime; you don’t need to download a local CSV.

---

## Problem Formulation

- **Task A (multiclass)**: predict **Arrest/Domestic** status  
  Classes (derived from `arrest` × `domestic`):
  - Non-Arrest Non-Domestic
  - Non-Arrest Domestic
  - Arrest Non-Domestic
  - Arrest Domestic

- **Task B (multiclass)**: predict **Primary Crime Type** (`primary_type`)  
  (singletons are removed to avoid classes with only one example).

**Features used for both tasks**
- `location_name` (derived from `block`, trimmed after the first 6 chars)
- `month` (from `date`)
- `hour_range` (string buckets like “08 am – 09 am”)
- `season` (1–4, computed from month)

All categorical fields are **LabelEncoded**; final feature set:
`[location_name_encoded, month, hour_range_encoded, season]`.

---

## Preprocessing (high level)

1. Drop rows with missing values.  
2. Parse `date` → add `month` & `hour`.  
3. Map hour → **hour_range** buckets (24 one-hour ranges).  
4. Compute **season** as `((month % 12) + 3) // 3`.  
5. Build `arrest_domestic` label from `arrest` & `domestic`.  
6. Keep relevant columns and label-encode categorical variables.  
7. Filter `primary_type` classes with only a single instance.

---

## Exploratory Analysis

- Histograms for encoded features.  
- Correlation heatmap across encoded variables.  
- Class distribution pie chart for `arrest_domestic_encoded`.

(Plots are generated with Matplotlib/Seaborn.)

---

## Model Training & Validation

Data is split **60% train / 20% validation / 20% test** (stratified):

- **Random Forest** (`n_estimators=100`, `max_depth=10`)
- **Decision Tree** (defaults)
- **XGBoost** (`n_estimators=800`, `learning_rate=0.1`)
- **KNN** (`n_neighbors=16`)

**Metrics reported** on validation:
- **Accuracy**
- **Weighted F1**  
(A test-set **confusion matrix** is shown for the Random Forest arrest model.)

### Cross-validation & Tuning
- 5-fold cross-validation scores printed for each model (on the validation split).  
- **RandomizedSearchCV** used to tune:
  - Random Forest: `n_estimators`, `max_depth`
  - KNN: `n_neighbors`, `weights`, `p`

> Note: For strict methodology you’d normally perform CV and tuning on the **train** split, then evaluate on val/test; the notebook keeps things simple and transparent.

---

## Outputs

- **Console metrics**: Accuracy & weighted F1 for all models on both tasks.
- **Confusion matrix**: for the Arrest/Domestic Random Forest on the test split.
- **`predictions.csv`**: model probabilities across **all** combinations of  
  `location_name × hour_range × month × season`, including:
  - probabilities for each Arrest/Domestic class
  - top-10 **primary_type** labels with associated probabilities

---

## How to Run

```bash
# open ML_final_project_group7.ipynb in Jupyter/Colab and run all cells
python ml_final_project_group7.py

