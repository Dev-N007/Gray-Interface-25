# Heart Disease Prediction Web App

This project focuses on building a machine learning-based web application to predict the likelihood of heart disease using the UCI Heart Disease dataset. Two classifiers — Random Forest and Support Vector Classifier (SVC) — were trained and evaluated. SHAP analysis was performed for model interpretability, and the model was deployed as a Flask web application.

## 📁 Dataset

- **Source**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **File Used**: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
- **Target**: Presence (1) or absence (0) of heart disease

## 📊 Model Evaluation

### 🔁 Random Forest Classifier

| Metric       | Class 0 | Class 1 |
|--------------|---------|---------|
| Precision    | 0.82    | 0.81    |
| Recall       | 0.76    | 0.86    |
| F1-Score     | 0.78    | 0.84    |
| Support      | 82      | 102     |

- **Overall Accuracy**: 82%
- **Macro Avg F1-Score**: 0.81
- **Weighted Avg F1-Score**: 0.81

---

### 💡 Support Vector Classifier (SVC)

| Metric       | Class 0 | Class 1 |
|--------------|---------|---------|
| Precision    | 0.82    | 0.85    |
| Recall       | 0.80    | 0.86    |
| F1-Score     | 0.81    | 0.85    |
| Support      | 82      | 102     |

- **Overall Accuracy**: 84%
- **Macro Avg F1-Score**: 0.83
- **Weighted Avg F1-Score**: 0.84

## 🔍 SHAP Analysis

- **Goal**: To interpret the model’s predictions by analyzing feature contributions.
- **Tool**: SHAP (SHapley Additive exPlanations)
- **Outcome**: Identified top contributing features for heart disease prediction, improving trust in model decisions.

## 🌐 Flask Web Application

- Developed a user-friendly Flask interface.
- Allows users to input medical parameters and receive prediction results.
- SHAP visualizations can optionally be displayed for interpretability.

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: pandas, scikit-learn, SHAP, Flask, matplotlib, seaborn
- **Deployment**: Flask server (locally runnable or deployable to services like Heroku)

## 🚀 How to Run

# Colab Link - https://colab.research.google.com/drive/1VFQCPfREfe129Iyf-YbAShDsaBTQAfRs?usp=sharing

```bash
# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py
