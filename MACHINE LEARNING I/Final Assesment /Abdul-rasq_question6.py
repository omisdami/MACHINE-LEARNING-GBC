import pandas as pd
import numpy as np
import warnings
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


warnings.filterwarnings("ignore")

def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(kernel='linear', probability=True),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
        }
    
    return metrics

def ensemble_learning(X_train, X_test, y_train, y_test, method="voting"):
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('svm', SVC(kernel='linear', probability=True)),
        ('lr', LogisticRegression(max_iter=1000))
    ]
    
    if method == "voting":
        ensemble_model = VotingClassifier(estimators=base_models, voting='soft')
    elif method == "stacking":
        ensemble_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(max_iter=1000))
    else:
        raise ValueError("Invalid ensemble method. Choose 'voting' or 'stacking'.")

    ensemble_model.fit(X_train, y_train)
    y_pred = ensemble_model.predict(X_test)
    y_prob = ensemble_model.predict_proba(X_test)[:, 1] if hasattr(ensemble_model, "predict_proba") else None
    
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
    }

def main(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df.drop_duplicates().copy()

    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    numeric_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_metrics = train_models(X_train, X_test, y_train, y_test)
    ensemble_metrics = ensemble_learning(X_train, X_test, y_train, y_test)

    print(f"Ensemble Model Metrics: {ensemble_metrics}")
    print("Model Metrics:", model_metrics)

main("https://raw.githubusercontent.com/subashgandyer/datasets/main/great_customers.csv")
