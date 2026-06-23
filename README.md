# Retention Time Predictor

## Requirements

* Python 3.10+
* A Google Gemini API Key

## Installation

### 1. Clone the project

git clone <repository_url>
cd <project_folder>



# 2. Install dependencies


pip install flask flask-cors python-dotenv google-generativeai pandas numpy openpyxl


## Gemini API Configuration

Create a file named `.env` in the project root directory:


API_KEY=YOUR_GEMINI_API_KEY


You can obtain a Gemini API key from:

https://aistudio.google.com/app/apikey

## Run the Application


python gemini_predictin_model.py


The application will start on:


http://127.0.0.1:5000


A browser window should open automatically.

## Usage

1. Upload an Excel dataset (`.xlsx`) and/or a scientific paper (`.pdf`).
2. Select the Excel sheet if a dataset was uploaded.
3. Wait for Gemini initialization.
4. Select a compound from the dataset or enter a SMILES manually.
5. Run the prediction and view the predicted retention time.

## Excel Requirements

The Excel file must contain:

* An RT column (`rt`, `rt_min`, or `retention_time`)
* A SMILES column (`smiles` or `pubchem.smiles.canonical`)

The dataset is automatically split into:

* 80% context data
* 20% prediction data
