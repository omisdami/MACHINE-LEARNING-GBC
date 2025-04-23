import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")

def load_data(url):
    df = pd.read_csv(url)
    df = df.drop_duplicates()
    df = df.dropna()
    print("Dropped all missing values")
    return df

def preprocess_data(df):
    df = df.iloc[:, 1:]  # Drop unique identifier column
    label_encoder = LabelEncoder()
    df['great_customer_class'] = label_encoder.fit_transform(df['great_customer_class'])
    
    categorical_features = ["workclass", "salary", "education_rank", "marital-status", "occupation", "race", "sex"]
    column_transformer = ColumnTransformer([
        ('one_hot', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ], remainder='passthrough')
    
    X = df.drop(columns=['great_customer_class'])
    y = df['great_customer_class']
    X_transformed = column_transformer.fit_transform(X)
    print("Applied one-hot encoding")
    
    return X_transformed, y

def feature_selection(X, y, k=5):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    print(f"Selected top {k} features")
    return X_new

def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        "SVM": SVC(kernel='rbf', C=0.5, probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
    }
    
    accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy
        print(f"Accuracy of {name}: {accuracy:.4f}")
    
    return models, accuracies

def ensemble_learning(models, X_train, X_test, y_train, y_test):
    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()], voting='hard'
    )
    voting_clf.fit(X_train, y_train)
    y_pred_ensemble = voting_clf.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    print(f"Accuracy of Ensemble Model (Voting Classifier): {ensemble_accuracy:.4f}")
    return y_pred_ensemble

def evaluate_model(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

def main():
    url = "https://raw.githubusercontent.com/subashgandyer/datasets/main/great_customers.csv"
    df = load_data(url)
    X, y = preprocess_data(df)
    X_selected = feature_selection(X, y, k=5)
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
    models, accuracies = train_models(X_train, X_test, y_train, y_test)
    y_pred_ensemble = ensemble_learning(models, X_train, X_test, y_train, y_test)
    evaluate_model(y_test, y_pred_ensemble)

if __name__ == "__main__":
    main()
