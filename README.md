# Amazon Sentiment Analysis

Machine learning sentiment analysis project with:
- Training and analysis notebook
- Flask API for real-time prediction
- Saved model artifacts for deployment

## Project Structure

- `api/app.py` - Flask backend and web UI
- `notebooks/Machine_Learning_Malai_Chaap_Final.ipynb` - End-to-end ML notebook
- `saved_models/` - Trained models and TF-IDF vectorizers
- `docs/Notebook_Full_Explanation.md` - Detailed notebook explanation

## Quick Start

### 1. Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\\Scripts\\activate    # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run API

From project root:

```bash
python api/app.py
```

Open: http://localhost:5000

Health check: http://localhost:5000/health

### 3. Run Notebook (optional)

Open and run:
- `notebooks/Machine_Learning_Malai_Chaap_Final.ipynb`

Model save/load paths are configured relative to notebook and API structure.

## Notes

- API loads `lr_clothing.pkl` and `tfidf_clothing.pkl` from `saved_models/`.
- If you retrain models in notebook, they are saved back to `saved_models/`.

## Git Commands

```bash
git init
git branch -M main
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```
