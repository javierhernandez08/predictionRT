# Retention Time Predictor

## Requirements

* Python 3.10+
* A Google Gemini API Key (or several, for automatic rotation)
* RDKit (optional, but recommended for accurate molecular fingerprints)

## Installation

### 1. Clone the project

git clone <repository_url>
cd <project_folder>

### 2. Install dependencies

pip install flask flask-cors python-dotenv google-genai pandas openpyxl numpy scikit-learn rdkit fpdf2 matplotlib

If RDKit fails to install on your system, the app will still run. It falls back to a pure-Python hashing fingerprint instead of RDKit's Morgan fingerprint. Prediction quality will be slightly lower in that case.

## IMPORTANT

For RDKit to import correctly on Windows, Smart App Control must be turned off. Otherwise the RDKit DLLs are blocked, the import fails silently and the fingerprints fall back to the hashing method. After disabling it, restart the computer before running the script. The console prints whether RDKit loaded correctly when the app starts.

| Package | Used for |
|---|---|
| flask | The web server (API routes, serving the interface) |
| flask-cors | Allowing CORS requests from the browser |
| python-dotenv | Loading the API keys from the .env file |
| google-genai | The Gemini API client |
| pandas | Reading the Excel file and handling the data tables |
| openpyxl | Engine pandas uses to read .xlsx files |
| numpy | Numerical computation (fingerprints, statistics, feature matrices) |
| scikit-learn | Linear Regression and Random Forest (scenarios E and F) |
| rdkit | Real molecular fingerprints from SMILES (scenarios E and F) and InChI to SMILES conversion in scenario H |
| fpdf2 | Generating the downloadable statistics PDF |
| matplotlib | Drawing the decision tree chart (scenario F) |
| pubchempy (optional) | Only needed for scenario H when a dataset gives compound NAMES instead of SMILES or InChI. Resolves names to structures via PubChem, so it needs internet access. Install with pip install pubchempy if you plan to use scenario H on name only datasets. |

Everything else the script imports (os, re, csv, time, hashlib, threading, webbrowser, io, typing) is part of Python's standard library and does not need to be installed separately.

## Gemini API Configuration

Create a file named .env in the project root directory:

GEMINI_API_KEY_1=YOUR_GEMINI_API_KEY
GEMINI_API_KEY_2=YOUR_SECOND_GEMINI_API_KEY
GEMINI_API_KEY_3=YOUR_THIRD_GEMINI_API_KEY

You can add up to 10 keys (GEMINI_API_KEY_1 through GEMINI_API_KEY_10). The app rotates between them automatically and puts a key on a short cooldown if it hits a rate limit or quota error, so you do not have to babysit it during long runs. A single key also works, using either GEMINI_API_KEY_1 or just GEMINI_API_KEY or API_KEY.

You can obtain a Gemini API key from:

https://aistudio.google.com/app/apikey

## Run the Application

python gemini_prediction_model.py

The application will start on:

http://127.0.0.1:5000

A browser window should open automatically.

## Usage

1. Choose one of the eight scenarios (A through H, see below).
2. For scenarios A to F, upload an Excel dataset (.xlsx or .xls). A scientific paper (.pdf) can also be uploaded for scenarios C and D (required for D, optional for C). For scenario G, the manual SMILES panel opens directly, and the Excel upload is only needed if you enable the optional 80% context. For scenario H, upload an external CSV or Excel dataset instead (see below).
3. For scenarios A to F and H, select the Excel sheet to work with.
4. Depending on the scenario, either run the prediction on that sheet's held-out 20%, type in a SMILES by hand and let Gemini predict its RT (scenario G, optionally with 80% context from a sheet), or run the 80/20 evaluation directly on the external dataset (scenario H).
5. Review the results, then check the accumulated statistics for that scenario (MAE, RMSE, R2, correlations, regression fit) and download them as CSV, PDF or a chart image if needed.

## Scenarios

| Scenario | Description | Engine |
|---|---|---|
| A | No context, only the SMILES of the 20% test split | Gemini |
| B | 80% context (RT plus SMILES plus conditions) predicts the 20% | Gemini |
| C | Same as B, plus an optional scientific paper | Gemini |
| D | No RT context, only SMILES from the 20% plus a required paper | Gemini |
| E | Linear Regression trained on the sheet's 80%, predicts the 20% | Local ML |
| F | Random Forest trained on the sheet's 80%, predicts the 20% | Local ML |
| G | Type in any SMILES by hand and Gemini predicts its RT, optionally with 80% context from a sheet | Gemini |
| H | Auto-detects the compound and RT columns of an external dataset (SMILES, InChI or compound name), then 80/20 evaluation | Local ML or Gemini |

Scenarios A, B, C, D and G call the Gemini API and need at least one key configured in .env. Scenarios E and F run entirely locally with scikit-learn and do not use the API or consume any quota. Scenario H can use either. Random Forest and Linear Regression run locally, while its optional LLM mode calls Gemini.

## Manual SMILES (Scenario G)

Scenario G works differently from the others. Instead of training a local model, it sends your typed in SMILES to Gemini together with an optional 80% context block. In the interface there is a checkbox "Use 80% context from an Excel sheet (optional)". If it is unchecked, Gemini receives only the SMILES and estimates the RT from the molecule's chemistry, same behaviour as scenario A. If it is checked, you can upload an Excel file directly from the same panel, pick one of its sheets, and the 80% training split of that sheet (RT plus SMILES plus any chromatographic condition columns) is sent to Gemini as context. Predictions made in scenario G are kept in a session list that can be downloaded as CSV, including whether context was used and from which sheet. Scenario G has no accumulated statistics panel, since it does not perform a held-out evaluation.

## External Datasets (Scenario H)

Scenario H is meant for datasets from other published studies, where the columns will not already match this app's naming convention. Instead of the usual Excel and paper upload, it accepts a single CSV or Excel file and automatically detects:

* The RT column. Tries rt, rt_min, retention_time, retention_time_corrected and a few other common variants.
* The compound column, in this order of preference:
  1. SMILES directly, if a column matching that is found.
  2. InChI, converted to SMILES locally with RDKit (no internet needed).
  3. Compound name (for example "ibuprofen"), resolved to SMILES via PubChem using pubchempy, which requires internet access and the optional pubchempy package (see Requirements above). If neither pubchempy nor a SMILES or InChI column is available, the app will tell you exactly what is missing instead of failing silently.

CSV files are read with automatic delimiter detection, so a semicolon separated file, common in some European datasets, works without extra configuration. If the uploaded Excel file has more than one sheet, a sheet selector appears.

Once the dataset is loaded, the panel shows what was detected (RT column, compound column type, how many compounds are valid, how many rows were dropped because the RT was not numeric or the structure could not be parsed) before you run anything.

You then choose the algorithm:

* Random Forest or Linear Regression. Run entirely locally, on the full 20% test split, however large it is.
* LLM (Gemini). Reuses the same 80% context and 20% prediction approach as scenario B. Because sending an enormous dataset (tens of thousands of compounds) to Gemini in one run is impractical, the prediction set is automatically capped to a random sample of 500 compounds when it is larger than that. The results will indicate if this happened.

Results from scenario H feed into the same statistics, CSV, PDF, PNG downloads and chart as any other scenario.

## Model Insights (Scenario F only)

After running scenario F (Random Forest) on a sheet, the statistics panel shows an extra download button so you can inspect the model itself, not just its accuracy:

* Download a tree (PNG). Draws one of the 300 trees in the Random Forest. Only the first 3 levels are shown, since the full tree is far too deep and large to read. The caption makes this clear.

This chart labels the 1024 Morgan fingerprint bits as fp_0, fp_1, and so on, since a single bit does not correspond to a readable chemical name on its own. Any named chromatographic condition columns present in the sheet (pH, temperature, flow rate, and so on) keep their real names.

The button only appears when the currently selected scenario is F and a Random Forest model has been trained on at least one sheet during the current session. It is not shown for scenarios E, G or H, even if those scenarios also train a model internally.

## Downloading Results

For scenarios A to F and H, once a scenario has accumulated predictions across one or more sheets, you can download:

* CSV. Raw predictions (SMILES, real RT, predicted RT, absolute error)
* PDF. A formatted report with all statistical metrics
* PNG. The predicted versus real scatter chart

Scenario F additionally offers the decision tree PNG download described in Model Insights above.

## Excel Requirements

The Excel file must contain:

* An RT column (rt, rt_min, or retention_time)
* A SMILES column (smiles or pubchem.smiles.canonical)

Optional columns recognised automatically (used as extra context where relevant): column.name, column.usp.code, column.length, column.particle.size, column.temperature, column.flowrate, eluent.A.h2o, eluent.A.pH, eluent.B.meoh, eluent.B.pH, gradient.start.A, gradient.end.B.

For scenarios A to F, each sheet is automatically split into:

* 80% context data
* 20% prediction data

Scenario G does not use this split. If you enable the context checkbox, it uses the same 80% training split of the chosen sheet as examples for Gemini. If you leave it disabled, no sheet is needed and only the SMILES is sent.
