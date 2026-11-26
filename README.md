# Horizon: a diabetes risk lens

Most risk calculators return one number. Horizon also shows:
- **Peer context**: how many of the 30 most similar patients in the dataset had diabetes.
- **What would move it**: how much your risk drops if glucose, BMI or blood pressure were lower, ranked by impact.
- **Missing-data handling**: zeros in this dataset mean "not measured", so they are treated as missing, not as real values.

Model: median imputation, scaling and logistic regression (5-fold CV: accuracy 77%, ROC-AUC 0.84, recall 57%).
Data: Pima Indians Diabetes dataset (768 women of Pima Indian heritage, 21+).

## Run
```
pip install -r requirements.txt
python train.py        # optional, model.pkl is included
streamlit run app.py
```
Deploy free on Streamlit Community Cloud: connect this repo, main file `app.py`.

## Limits
Not a diagnosis. The data covers one population, and the "what would move it" figures are associations in that data, not proven effects.
