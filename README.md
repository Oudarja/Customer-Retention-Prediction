# Customer Retention Prediction

Predict whether a customer will churn based on their previous data using Machine Learning and Deep Learning models. This project includes a complete ML pipeline with EDA, preprocessing, feature engineering, model training, evaluation, and visualization.

---

## Dataset

- **Columns:**
  - `Customer_ID`: Unique customer identifier
  - `Age`: Customer age
  - `Gender`: Male/Female
  - `Annual_Income`: Yearly income
  - `Total_Spend`: Total spending till now
  - `Years_as_Customer`: Duration with company
  - `Num_of_Purchases`: Number of purchases
  - `Average_Transaction_Amount`: Avg amount per transaction
  - `Num_of_Returns`: Number of returns
  - `Num_of_Support_Contacts`: Support interactions
  - `Satisfaction_Score`: Customer satisfaction rating
  - `Last_Purchase_Days_Ago`: Days since last purchase
  - `Email_Opt_In`: Whether opted in for emails
  - `Promotion_Response`: Response to promotions
  - `Target_Churn`: Target column (True if churned, False otherwise)

- **Target distribution:** Balanced

## Setup Instructions
### python version: 3.9

1. **Clone the repository**
```bash
git clone https://github.com/Oudarja/Customer-Retention-Prediction.git
cd Customer-Retention-Prediction
```
2. **Create virtual environment**
   ```
   python -m venv .venv
   ```
3. ***Activate virtual environment***
   ```
   windows: .venv\Scripts\activate
   Linux/MacOS: source .venv/bin/activate
   ```
4. ***Install dependencies***
   ```
   pip install -r requirements.txt
   ```
6. ***Run evaluation script***
   ```
   python main.py
   ```

---
### Models Included
 - Random Forest
 - XGBoost
 - LightGBM
 - Artificial Neural Network (ANN)
#### Each model is trained using train/validation/test split and GridSearchCV for hyperparameter tuning. Feature selection process also has been used according to each model.
---
### Evaluation Metrics
 - Accuracy
 - Precision, Recall, F1-Score
 - ROC-AUC Score
 - Confusion Matrix
 - ROC Curve
---
### Key Results
- Compare models using metrics and ROC curves.
- Analyze which model performs best for customer churn prediction.
