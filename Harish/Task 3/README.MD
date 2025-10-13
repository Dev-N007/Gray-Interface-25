# Personality Classification using SHAP: Introvert vs Extrovert

<h2>Kaggle Link:</h2>

 Visit [Introvert_Vs_Extrovert](https://www.kaggle.com/code/harishsinghbhadauria/introvert-vs-extrovert) to see the my Kaggle notebook.


## 📅 Project Overview

This project focuses on the classification of human personalities as **Introvert** or **Extrovert** based on behavioral and psychological traits. The central goal is not only to **accurately predict personality types**, but also to ensure **transparent model behavior** using **SHAP (SHapley Additive exPlanations)**.

We evaluate multiple machine learning models and focus on how SHAP values can help us:

* Understand which features influence predictions.
* Explain individual predictions.
* Choose the most interpretable and reliable model.

---

## 📊 Dataset Description

The dataset consists of responses to questions capturing behavioral tendencies, such as:

* `Stage_fear`: Does the person fear public speaking?
* `Drained_after_socializing`: Does the person feel mentally drained after interacting with people?
* `Personality`: Target label (Extrovert or Introvert)

Categorical features are ordinally encoded to allow proper model training.

---

## 💡 Why SHAP?

While accuracy is critical, in sensitive domains like personality prediction, **explainability is equally important**. SHAP helps us answer key questions:

> “Why did the model predict that this person is an introvert?”
> “Which traits contribute most to personality classification?”

### SHAP Advantages:

* **Global explanations**: Feature importance across the dataset.
* **Local explanations**: How individual features affect each prediction.
* **Fair & additive**: Based on cooperative game theory.

---

## 🎓 Models Compared

Two machine learning models were trained and evaluated:

| Model                           | Type         | Explainability Support  |
| ------------------------------- | ------------ | ----------------------- |
| Random Forest Classifier        | Tree-based   | ✅ Excellent with SHAP   |
| Support Vector Classifier (SVC) | Kernel-based | ❌ Approximate with SHAP |

---

## 📈 SHAP Analysis and Visualizations

### 📊 1. Random Forest Classifier

#### ▶ SHAP Summary Plot

* **Top features**: `Stage_fear`, `Drained_after_socializing`
* **Interpretation**: High `Stage_fear` and `Drained_after_socializing` increase the likelihood of being classified as an introvert.

> 📅 **Insight**: This plot ranks features by average contribution. Color gradients (red = high, blue = low) reveal how values shift predictions. A global understanding of the model is achieved.

#### ▶ SHAP Waterfall Plot

* **Focus**: A single prediction.
* Shows how the base value is adjusted feature-by-feature to reach the final probability.

> 📅 **Insight**: Very useful for justifying decisions for individual predictions (e.g., in educational or mental health contexts).

---

### 📊 2. Support Vector Classifier (SVC)

#### ▶ SHAP Beeswarm Plot

* Gives a sense of feature importance and how values vary across predictions.

> 📅 **Insight**: Provides global distribution but harder to interpret for kernel-based methods.

#### ▶ SHAP Waterfall Plot

* Local explanation for one instance.

> ❌ **Limitation**: Since SVC is not tree-based, SHAP explanations are approximate, and contributions are not as intuitive.

---

## 🔝 Model Recommendation

### 🏆 Winner: **Random Forest Classifier**

* **Superior interpretability** with SHAP.
* Clearly identifies influential features.
* Performs well both in accuracy and explainability.

SVC, while useful, does not provide transparent feature contribution paths, which is critical for a task like personality analysis.

---

## 📄 Conclusion

This project demonstrates the importance of not only building **accurate predictive models** but also ensuring that those models can be **trusted and explained**. SHAP bridges this gap effectively, especially with tree-based models like Random Forest.

---

## 📂 Files in this Repository

* `Introvert_Vs_Extrovert.ipynb`: Jupyter Notebook with model training and SHAP analysis.
* `personality_datasert.csv`: Input dataset.
* `random_model.pkl`, etc.: Saved models.
* `svc_model.pkl`, etc.: Saved models.

---
