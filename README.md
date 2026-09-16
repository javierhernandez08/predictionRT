
A browser window should open automatically.

## Usage

1. Choose one of the eight scenarios (A through H, see below).
2. For scenarios A-G: upload an Excel dataset (`.xlsx`/`.xls`). A scientific paper (`.pdf`) can also be uploaded for scenarios C and D (required for D, optional for C). For scenario H: upload an external CSV or Excel dataset instead (see below).
3. For scenarios A-G, select the Excel sheet to work with. While the sheet is being loaded, a spinner is shown and the sheet info (valid rows, 80/20 split, whether it was already processed) appears only once it has finished; the run button stays disabled until then.
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

## Model Insights (Scenario F only)

After running scenario F (Random Forest) on a sheet, the statistics panel shows an extra download button so you can inspect the model itself, not just its accuracy:

* **🌳 Download a tree (PNG)** — draws one of the 300 trees in the Random Forest. Only the first 3 levels are shown, since the full tree is far too deep and large to read; the caption makes this clear.

This chart labels the 1024 Morgan fingerprint bits as `fp_0`, `fp_1`, etc., since a single bit does not correspond to a readable chemical name on its own. Any named chromatographic condition columns present in the sheet (pH, temperature, flow rate, and so on) keep their real names.

The button only appears when the currently selected scenario is **F** and a Random Forest model has been trained on at least one sheet during the current session. It is not offered for scenarios E, G or H, even if those scenarios also trained a model internally.

## Downloading Results

For scenarios A-F and H, once a scenario has accumulated predictions across one or more sheets, you can download:

* **CSV** — raw predictions (SMILES, real RT, predicted RT, absolute error)
* **PDF** — a formatted report with all statistical metrics
* **PNG** — the predicted-vs-real scatter chart

Scenario F additionally offers the decision-tree PNG download described in [Model Insights](#model-insights-scenario-f-only) above.

## Excel Requirements

The Excel file must contain:

* An RT column (`rt`, `rt_min`, or `retention_time`)
* A SMILES column (`smiles` or `pubchem.smiles.canonical`)

Optional columns recognised automatically (used as extra context where relevant): `column.name`, `column.usp.code`, `column.length`, `column.particle.size`, `column.temperature`, `column.flowrate`, `eluent.A.h2o`, `eluent.A.pH`, `eluent.B.meoh`, `eluent.B.pH`, `gradient.start.A`, `gradient.end.B`.

For scenarios A-F, each sheet is automatically split into:

* 80% context data
* 20% prediction data

Scenario G ignores this split and trains on 100% of the sheet's valid rows.
