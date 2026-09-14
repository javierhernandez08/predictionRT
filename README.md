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
pip install flask flask-cors python-dotenv google-genai pandas openpyxl numpy scikit-learn rdkit fpdf2
```

If RDKit fails to install on your system, the app will still run: it falls back to a pure-Python hashing fingerprint instead of RDKit's Morgan fingerprint. Prediction quality will be slightly lower in that case.

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

1. Choose one of the seven scenarios (A through G, see below).
2. Upload an Excel dataset (`.xlsx`/`.xls`). A scientific paper (`.pdf`) can also be uploaded for scenarios C and D (required for D, optional for C).
3. Select the Excel sheet to work with.
4. Depending on the scenario, either run the prediction on that sheet's held-out 20%, or train a model on the whole sheet and predict a SMILES you type in by hand (scenario G).
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

Scenarios A-D call the Gemini API and need at least one key configured in `.env`. Scenarios E, F and G run entirely locally with scikit-learn and do not use the API or consume any quota.

## Manual SMILES (Scenario G)

Scenario G works differently from the others: instead of holding out 20% of a sheet, it trains a model (Random Forest or Linear Regression, selectable in the interface) on every valid row of the chosen sheet, then lets you type in any SMILES string and get an instant predicted RT. If the sheet has chromatographic condition columns (column type, pH, flow rate, gradient, etc.), you can optionally fill those in too; anything left blank defaults to 0. Predictions made this way are kept in a session list that can be downloaded as CSV.

## Downloading Results

For scenarios A-F, once a scenario has accumulated predictions across one or more sheets, you can download:

* **CSV** - raw predictions (SMILES, real RT, predicted RT, absolute error)
* **PDF** - a formatted report with all statistical metrics
* **PNG** - the predicted-vs-real scatter chart

## Excel Requirements

The Excel file must contain:

* An RT column (`rt`, `rt_min`, or `retention_time`)
* A SMILES column (`smiles` or `pubchem.smiles.canonical`)

Optional columns recognised automatically (used as extra context where relevant): `column.name`, `column.usp.code`, `column.length`, `column.particle.size`, `column.temperature`, `column.flowrate`, `eluent.A.h2o`, `eluent.A.pH`, `eluent.B.meoh`, `eluent.B.pH`, `gradient.start.A`, `gradient.end.B`.

For scenarios A-F, each sheet is automatically split into:

* 80% context data
* 20% prediction data

Scenario G ignores this split and trains on 100% of the sheet's valid rows.
