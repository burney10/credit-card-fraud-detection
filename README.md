# 💳 Credit Card Fraud Detection
### End-to-End Machine Learning Pipeline for Imbalanced Classification

---

## 📌 Project Overview
This project builds a **production-oriented credit card fraud detection system** using highly imbalanced transaction data.  
The focus is on **handling extreme class imbalance, correct metric selection, model comparison, explainability, and threshold optimization**.

---

## 📊 Dataset
- **Source:** Kaggle Credit Card Fraud Dataset  
- **Total transactions:** 284,807  
- **Fraud cases:** 492 (≈ 0.17%)  
- **Features:**
  - `V1–V28`: PCA-transformed features  
  - `Time`, `Amount`

> Due to confidentiality, original feature descriptions are not available.

---

## 📥 Dataset Access
Download the dataset from Kaggle:
https://www.kaggle.com/mlg-ulb/creditcardfraud

---

## ⚠️ Key Challenges
- Severe class imbalance  
- Accuracy is misleading for evaluation  
- High cost of false negatives (missed frauds)  
- Need for explainable predictions  

---

## 🧠 Techniques Used
- **Imbalance Handling:** SMOTE, class weighting  
- **Models Implemented:**
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
  - LightGBM  
  - Isolation Forest  
- **Evaluation Metrics:** PR-AUC (primary), ROC-AUC  
- **Explainability:** SHAP  
- **Optimization:** Classification threshold tuning  

---

## 🏆 Best Model Performance

| Model     | Technique | PR-AUC |
|----------|-----------|--------|
| LightGBM | SMOTE     | **0.8569** |

- High fraud recall  
- Controlled false positives  
- Interpretable predictions using SHAP  

---

## 🔍 Explainability
SHAP is used to:
- Identify influential PCA components  
- Explain individual fraud predictions  
- Improve transparency and trust  

---

## 🎯 Threshold Optimization
Instead of using the default `0.5` threshold, the model is tuned to:
- Increase fraud detection rate  
- Balance recall–precision trade-off  
- Reduce business risk  

---

## ▶️ How to Run
```bash```
pip install -r requirements.txt  then 
jupyter notebook notebooks/01_fraud_detection.ipynb

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn
- XGBoost, LightGBM
- SHAP
- Matplotlib, Seaborn

---

## 🎓 Skills Demonstrated

- Handling highly imbalanced datasets
- Correct metric selection for classification
- Model comparison & evaluation
- Explainable AI (XAI)
- Threshold tuning for business optimization
- End-to-end ML pipeline design

---

## 📌 Final Note

This project reflects real-world ML decision-making, focusing not just on accuracy but on impact, interpretability, and deployment readiness.
