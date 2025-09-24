# 🚢 Titanic Survival Prediction – Machine Learning Project

A complete machine learning pipeline built using the Titanic dataset to predict passenger survival. This project includes data exploration, preprocessing, model training, testing, and evaluation. It serves as a foundational project for understanding classification problems in data science.

---

## 📂 Project Structure

```
titanic-survival-prediction/
│
├── data/
│   └── train.csv
│   └── test.csv
│
├── notebooks/
│   └── titanic_model.ipynb     # Full EDA, preprocessing, training
│
├── models/
│   └── model.pkl               # Trained ML model (optional)
│
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── utils.py                    # Optional utility functions
```

---

## 📊 Dataset Overview

The Titanic dataset contains demographic and travel information of passengers aboard the Titanic. Key features include:

* **PassengerId**
* **Pclass** (Ticket class)
* **Sex**
* **Age**
* **SibSp** (Siblings/Spouses aboard)
* **Parch** (Parents/Children aboard)
* **Fare**
* **Embarked**
* **Survived** *(Target variable: 0 = No, 1 = Yes)*

Dataset Source: [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic)

---

## 🧰 Technologies Used

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* Matplotlib & Seaborn
* Jupyter Notebook / vscode(optional)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/RM7402704/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Notebook

```bash
jupyter notebook notebooks/titanic_model.ipynb
```

---

## 🧪 Project Workflow

### ✅ Step 1: Exploratory Data Analysis (EDA)

* Checked for null values and data types
* Visualized distributions of age, class, fare, etc.
* Analyzed survival rates across different groups (e.g., sex, class, age)

📌 **Key Insight:** Women and passengers in higher classes had higher survival rates.

---

### ✅ Step 2: Data Preprocessing

* Handled missing values (e.g., imputed age with median)
* Converted categorical variables (e.g., 'Sex', 'Embarked') using one-hot encoding
* Scaled numerical features
* Split data into training and test sets

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### ✅ Step 3: Model Building & Training

Tested multiple classification models including:

* **Logistic Regression**
* **Random Forest**
* **Support Vector Machine (SVM)**
* **K-Nearest Neighbors (KNN)**
* **Decision Tree**

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
```

---

### ✅ Step 4: Model Evaluation

Evaluated model performance using:

* Accuracy Score
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

📌 **Best Model:** *Random Forest* performed best with highest accuracy and balanced performance across classes.

---

## 📈 Sample Output

```text
Accuracy: 0.81

              precision    recall  f1-score   support

           0       0.85      0.85      0.85       105
           1       0.75      0.75      0.75        74

    accuracy                           0.81       179
```

---

## 💾 Model Deployment (Optional)

* Saved the trained model using `joblib` or `pickle`
* Can be integrated into a web app using Flask or Streamlit for real-time predictions

```python
import joblib
joblib.dump(model, 'models/titanic_model.pkl')
```

---

## 🔑 Key Takeaways

* **Gender and class** are strong predictors of survival.
* Handling missing values and encoding categorical features are crucial steps.
* Ensemble models (like Random Forest) often outperform simple classifiers.

---

## 🧠 Future Improvements

* Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
* Cross-validation for more robust evaluation
* Deploy model via API or web interface

---

## 🤝 Contributing

Contributions, issues, and suggestions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

* [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic)
* [Scikit-learn Documentation](https://scikit-learn.org/)

---
