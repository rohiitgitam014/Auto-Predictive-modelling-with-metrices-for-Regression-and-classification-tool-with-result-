# Auto-Predictive-modelling-with-metrices-for-Regression-and-classification-tool-with-result-
🎯 Objective
The primary goal of this AutoML tool is to simplify the machine learning process for non-technical users by enabling automatic model training and evaluation for both classification and regression problems. Users can upload a dataset, select the target column, and receive instant model performance results without needing to write code.
🛠️ Key Features
1. User-Friendly Web Interface: Built using Streamlit for real-time interaction.
2. Automatic Task Detection:
   - Classification if the target variable has fewer than 20 unique values and is not float.
   - Regression otherwise.
3. Data Cleaning:
   - Automatically replaces empty strings with NaN and removes missing values.
4. Label Encoding:
   - Encodes categorical features to numeric format for model compatibility.
5. Model Training and Evaluation:
   - Classification Models:
     - Logistic Regression
     - Random Forest Classifier
     - Support Vector Machine (SVM)
   - Regression Models:
     - Linear Regression
     - Random Forest Regressor
     - Support Vector Regressor (SVR)
📊 Evaluation Metrics
- Classification:
  - Accuracy
  - F1 Score
  - Recall
  - Precision

- Regression:
  - R² Score
  - RMSE (Root Mean Squared Error)
  - MSE (Mean Squared Error)
📈 Workflow
1. User uploads a CSV file.
2. The dataset is previewed and cleaned.
3. The user selects a target column.
4. The tool determines the task type (classification/regression).
5. Multiple models are trained and tested using an 80-20 train/test split.
6. Results are displayed in a ranked, sortable table.
✅ Conclusion
The AutoML Tool developed using Streamlit delivers a powerful and intuitive solution for quickly applying machine learning to structured datasets. It enables users—regardless of technical background—to build, compare, and evaluate predictive models with minimal effort. By automating key ML processes such as preprocessing, task detection, and metric reporting, the tool supports faster decision-making and broader adoption of data science practices within business environments.
