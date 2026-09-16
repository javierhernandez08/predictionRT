# Retention Time Predictor

## Requirements

* Python 3.10+
* A Google Gemini API Key (or several, for automatic rotation)
* RDKit (optional, but recommended for accurate molecular fingerprints)

## Installation

### 1. Clone the project

```
git clone <repository_url>
cd <project_folder>
```

### 2. Install dependencies

```
pip install flask flask-cors python-dotenv google-genai pandas openpyxl numpy scikit-learn rdkit fpdf2 matplotlib
```

If RDKit fails to install on your system, the app will still run: it falls back to a pure-Python hashing fingerprint instead of RDKit's Morgan fingerprint. Prediction quality will be slightly lower in that case.

| Package | Used for |
|---|---|
| `flask` | The web server (API routes, serving the interface) |
| `flask-cors` | Allowing CORS requests from the browser |
| `python-dotenv` | Loading the API keys from the `.env` file |
| `google-genai` | The Gemini API client |
| `pandas` | Reading the Excel file and handling the data tables |
| `openpyxl` | Engine pandas uses to read `.xlsx` files |
| `numpy` | Numerical computation (fingerprints, statistics, feature matrices) |
| `scikit-learn` | Linear Regression and Random Forest (scenarios E, F and G) |
| `rdkit` | Real molecular fingerprints from SMILES (scenarios E, F and G), and InChI-to-SMILES conversion in scenario H |
| `fpdf2` | Generating the downloadable statistics PDF |
| `matplotlib` | Drawing the decision tree, feature importance and coefficient charts (scenarios E and F) |
| `pubchempy` (optional) | Only needed for scenario H when a dataset gives compound NAMES instead of SMILES/InChI. Resolves names to structures via PubChem, so it needs internet access. Install with `pip install pubchempy` if you plan to use scenario H on name-only datasets. |

Everything else the script imports (`os`, `re`, `csv`, `time`, `hashlib`, `threading`, `webbrowser`, `io`, `typing`) is part of Python's standard library and does not need to be installed separately.

## Gemini API Configuration

Create a file named `.env` in the project root directory:

```
GEMINI_API_KEY_1=YOUR_GEMINI_API_KEY
GEMINI_API_KEY_2=YOUR_SECOND_GEMINI_API_KEY
GEMINI_API_KEY_3=YOUR_THIRD_GEMINI_API_KEY
```

You can add up to 10 keys (`GEMINI_API_KEY_1` through `GEMINI_API_KEY_10`). The app rotates between them automatically and puts a key on a short cooldown if it hits a rate limit or quota error, so you do not have to babysit it during long runs. A single key also works, using either `GEMINI_API_KEY_1` or just `GEMINI_API_KEY` / `API_KEY`.

You can obtain a Gemini API key from:

https://aistudio.google.com/app/apikey

## Run the Application

```
python gemini_prediction_model.py
```

The application will start on:

```
http://127.0.0.1:5000
```

A browser window should open automatically.

## Usage

1. Choose one of the eight scenarios (A through H, see below).
2. For scenarios A-G: upload an Excel dataset (`.xlsx`/`.xls`). A scientific paper (`.pdf`) can also be uploaded for scenarios C and D (required for D, optional for C). For scenario H: upload an external CSV or Excel dataset instead (see below).
3. For scenarios A-G, select the Excel sheet to work with.
4. Depending on the scenario, either run the prediction on that sheet's held-out 20%, train a model on the whole sheet and predict a SMILES you type in by hand (scenario G), or run the 80/20 evaluation directly on the external dataset (scenario H).
5. Review the results, then check the accumulated statistics for that scenario (MAE, RMSE, R², correlations, regression fit) and download them as CSV, PDF or a chart image if needed.

## Scenarios

| Scenario | Description | Engine |
|---|---|---|
| A | No context, only the SMILES of the 20% test split | Gemini |
| B | 80% context (RT + SMILES + conditions) predicts the 20% | Gemini |
| C | Same as B, plus an optional scientific paper | Gemini |
| D | No RT context, only SMILES from the 20% + a required paper | Gemini |
| E | Linear Regression trained on the sheet's 80%, predicts the 20% | Local ML |
| F | Random Forest trained on the sheet's 80%, predicts the 20% | Local ML |
| G | Train on the whole sheet, then predict any SMILES typed in by hand | Local ML |
| H | Auto-detects the compound/RT columns of an external dataset (SMILES, InChI or compound name), then 80/20 evaluation | Local ML or Gemini |

Scenarios A-D call the Gemini API and need at least one key configured in `.env`. Scenarios E, F and G run entirely locally with scikit-learn and do not use the API or consume any quota. Scenario H can use either: Random Forest and Linear Regression run locally, while its optional LLM mode calls Gemini.

## Manual SMILES (Scenario G)

Scenario G works differently from the others: instead of holding out 20% of a sheet, it trains a model (Random Forest or Linear Regression, selectable in the interface) on every valid row of the chosen sheet, then lets you type in any SMILES string and get an instant predicted RT. If the sheet has chromatographic condition columns (column type, pH, flow rate, gradient, etc.), you can optionally fill those in too; anything left blank defaults to 0. Predictions made this way are kept in a session list that can be downloaded as CSV.

## External Datasets (Scenario H)

Scenario H is meant for datasets from other published studies, where the columns will not already match this app's naming convention. Instead of the usual Excel + paper upload, it accepts a single **CSV or Excel** file and automatically detects:

* **The RT column** — tries `rt`, `rt_min`, `retention_time`, `retention_time_corrected` and a few other common variants.
* **The compound column**, in this order of preference:
  1. **SMILES** directly, if a column matching that is found.
  2. **InChI** — converted to SMILES locally with RDKit (no internet needed).
  3. **Compound name** (e.g. "ibuprofen") — resolved to SMILES via PubChem using `pubchempy`, which **requires internet access** and the optional `pubchempy` package (see Requirements above). If neither `pubchempy` nor a SMILES/InChI column is available, the app will tell you exactly what is missing instead of failing silently.

CSV files are read with automatic delimiter detection (so a `;`-separated file, common in some European datasets, works without extra configuration). If the uploaded Excel file has more than one sheet, a sheet selector appears.

Once the dataset is loaded, the panel shows what was detected (RT column, compound column type, how many compounds are valid, how many rows were dropped because the RT wasn't numeric or the structure couldn't be parsed) before you run anything.

You then choose the algorithm:

* **Random Forest** or **Linear Regression** — run entirely locally, on the full 20% test split, however large it is.
* **LLM (Gemini)** — reuses the same 80% context / 20% prediction approach as scenario B. Because sending an enormous dataset (tens of thousands of compounds) to Gemini in one run is impractical, the prediction set is automatically capped to a random sample of 500 compounds when it is larger than that; the results will indicate if this happened.

Results from scenario H feed into the same statistics, CSV/PDF/PNG downloads and chart as any other scenario.

## Model Insights (Scenarios E and F)

After running scenario E (Linear Regression) or F (Random Forest) on a sheet, the statistics panel shows extra download buttons so you can inspect the model itself, not just its accuracy:

* **🌳 Download a tree (PNG)** *(scenario F only)* — draws one of the 300 trees in the Random Forest. Only the first 3 levels are shown, since the full tree is far too deep and large to read; the caption makes this clear.
* **📊 Download feature importances (PNG)** *(scenario F only)* — a bar chart of the 20 features the forest relies on most.
* **📈 Download coefficients (PNG)** *(scenario E only)* — a bar chart of the Linear Regression coefficients, plus the intercept.

These charts label the 1024 Morgan fingerprint bits as `fp_0`, `fp_1`, etc., since a single bit does not correspond to a readable chemical name on its own. Any named chromatographic condition columns present in the sheet (pH, temperature, flow rate, and so on) keep their real names. For that reason, the coefficients chart for scenario E only plots the named columns and skips the 1024 fingerprint-bit coefficients, since those are not individually interpretable.

These buttons only appear once a model has actually been trained for that scenario in the current session (that is, after running at least one sheet with E or F). If you switch to a different scenario and come back, the buttons reflect whichever sheet you ran most recently.

## Downloading Results

For scenarios A-F and H, once a scenario has accumulated predictions across one or more sheets, you can download:

* **CSV** — raw predictions (SMILES, real RT, predicted RT, absolute error)
* **PDF** — a formatted report with all statistical metrics
* **PNG** — the predicted-vs-real scatter chart

Scenarios E and F additionally offer the model-specific PNG downloads described in [Model Insights](#model-insights-scenarios-e-and-f) below.

## Excel Requirements

The Excel file must contain:

* An RT column (`rt`, `rt_min`, or `retention_time`)
* A SMILES column (`smiles` or `pubchem.smiles.canonical`)

Optional columns recognised automatically (used as extra context where relevant): `column.name`, `column.usp.code`, `column.length`, `column.particle.size`, `column.temperature`, `column.flowrate`, `eluent.A.h2o`, `eluent.A.pH`, `eluent.B.meoh`, `eluent.B.pH`, `gradient.start.A`, `gradient.end.B`.

For scenarios A-F, each sheet is automatically split into:

* 80% context data
* 20% prediction data

Scenario G ignores this split and trains on 100% of the sheet's valid rows.
