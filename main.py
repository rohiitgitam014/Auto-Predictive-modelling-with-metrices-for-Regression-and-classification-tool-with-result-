import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error,recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

st.set_page_config(page_title="AutoML Tool", layout="wide")
st.title("🤖 AutoML Tool for Regression & Classification")
st.write("Upload a dataset and get model performance instantly!")

# Upload data
file = st.file_uploader("📁 Upload CSV File", type=["csv"])
if file:
    df = pd.read_csv(file)
    
    # Show raw preview
    st.write("### 📊 Raw Data Preview", df.head())

    # Data cleaning: remove empty strings and NaNs
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)

    st.success(f"✅ Cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    target_col = st.selectbox("🎯 Select the target column", df.columns)

    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode categorical features
        for col in X.select_dtypes(include='object').columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # Detect task type
        task = 'classification' if len(np.unique(y)) < 20 and y.dtype != 'float' else 'regression'
        st.info(f"🔍 Detected Task: **{task.title()}**")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        results = []

        if task == 'classification':
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest Classifier ": RandomForestClassifier(),
                "SVM": SVC()
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average='weighted')
                recall = recall_score(y_test, preds)
                precision = precision_score(y_test, preds)
                results.append((name, acc, f1, recall, precision))

            st.subheader("📈 Classification Results")
            results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Score", "Recall score", "Precision score"])
            st.dataframe(results_df.sort_values(by="Accuracy", ascending=False))

        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(),
                "SVR": SVR()
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                mse = mean_squared_error(y_test, preds)
                results.append((name, r2, rmse, mse))

            st.subheader("📉 Regression Results")
            results_df = pd.DataFrame(results, columns=["Model", "R2 Score", "RMSE", "MSE"])
            st.dataframe(results_df.sort_values(by="R2 Score", ascending=False))
