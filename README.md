# Titanic Survival Prediction

A beginner machine learning project that predicts whether a Titanic passenger survived, using binary classification.

## Overview

The sinking of the RMS Titanic in 1912 resulted in the loss of 1,502 lives. This project uses passenger data (age, gender, ticket class, etc.) to train and compare two classification models and predict survival outcomes.

**Dataset:** Built-in Titanic dataset from the `seaborn` library — no manual download required.

## Project Structure

```
Titanic_Survival_Prediction/
├── titanic_survival.ipynb  # Main notebook
├── requirements.txt
└── .gitignore
```

## Notebook Walkthrough

Both notebooks follow the same 6-step pipeline:

1. **Import Libraries** — NumPy, Pandas, Matplotlib, Seaborn, scikit-learn
2. **Load & Explore the Data** — dataset shape, info, missing value counts, survival breakdown, and visualizations (survival by gender and ticket class)
3. **Clean the Data** — keep relevant columns (`pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`), fill missing `age` with median, fill missing `embarked` with mode
4. **Prepare Features** — label-encode `sex` and `embarked`, engineer `family_size` feature, 80/20 train/test split
5. **Train Models** — Logistic Regression and Random Forest (100 estimators)
6. **Evaluate Models** — side-by-side confusion matrices, full classification reports, accuracy comparison bar chart, and Random Forest feature importance plot

## Results

Both models achieve ~80% accuracy on the test set. Key findings:
- **Sex** and **Fare** are the strongest predictors of survival
- Women and first-class passengers had significantly higher survival rates
- Random Forest slightly outperforms Logistic Regression

## Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/Titanic_Survival_Prediction.git
cd Titanic_Survival_Prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

Open `titanic_survival.ipynb` and run all cells.

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

## License

This project is open-source and available under the [MIT License](LICENSE).
