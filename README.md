# Interactive Web-Based Learning System for Logistic Regression

A Flask + scikit-learn web app that teaches Logistic Regression interactively.
Upload a dataset, preprocess, explore, train, evaluate, and predict — with
step-by-step mathematical explanations, visualizations, and an "Ask AI" tutor.

## Features

1. **Dataset Upload** — CSV upload, preview, shape, dtypes, missing values.
2. **Preprocessing** — missing-value imputation (mean/median/mode/drop),
   label / one-hot encoding, StandardScaler, before-vs-after comparison.
3. **EDA** — histograms, correlation heatmap, class distribution, feature-vs-target boxplots.
4. **Logistic Regression Learning Module** — visual + mathematical walkthrough:
   linear combination z = w·x + b, sigmoid σ(z), decision boundary, binary &
   multiclass (One-vs-Rest).
5. **Training Configuration** — train/test split slider, k-fold cross-validation.
6. **Model Training & Visualization** — learned weights & bias, decision
   boundary plot, sigmoid curve.
7. **Prediction Module** — input form for new data, shows z, σ(z), final class.
8. **Evaluation** — confusion matrix, accuracy, precision, recall, F1, ROC + AUC,
   CV results, full classification report.
9. **Educational Layer** — tooltips, side-panel explanations.
10. **Ask AI (Bonus)** — optional OpenAI integration (set `OPENAI_API_KEY`),
    with rule-based fallback if no key is configured.

## Project Structure

```
logreg_app/
├── app.py                     # Flask backend (all API endpoints)
├── requirements.txt
├── README.md
├── sample_data/
│   └── diabetes.csv           # Sample binary-classification dataset
├── templates/
│   └── index.html             # Single-page dashboard
└── static/
    ├── css/style.css
    └── js/app.js
```

## Setup

```bash
# 1. Create & activate a virtualenv (recommended)
python -m venv venv
source venv/bin/activate            # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Enable real LLM for "Ask AI"
export OPENAI_API_KEY=sk-...        # on Windows: set OPENAI_API_KEY=sk-...

# 4. Run
python app.py
```

Open http://localhost:5000 in your browser.

## Usage

1. Click **Load sample dataset** (or upload your own CSV).
2. Pick the **target column** and preprocessing options → **Apply preprocessing**.
3. Open **EDA** to inspect distributions and correlations.
4. Set the **train/test split** + **k folds** → **Train model**.
5. Inspect weights, decision boundary, and step-by-step sample explanation.
6. Use **Predict** to score a new input row.
7. Open **Evaluate** for confusion matrix, ROC, and full metrics.
8. Use **Ask AI** to ask conceptual questions.

## Notes

- State is stored in-memory per browser session. Restart the server to reset.
- For best decision-boundary visualization, use datasets with 2+ informative features.
- Multiclass uses One-vs-Rest by default.

## License

MIT — for educational use.
