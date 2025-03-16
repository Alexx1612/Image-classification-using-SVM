import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def optimize_svm_hyperparameters():
    """
    Hyperparameter optimization pipeline for SVM using Grid Search.
    """
    # Ensure required directories exist (create if needed)
    os.makedirs("models", exist_ok=True)

    # Load data (train.csv)
    print("Loading training data for Grid Search...")
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")["Label"]

    # Spliting data
    print("Creating train/validation split (80/20)...")
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print("Standardizing features...")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_split)
    x_val_scaled = scaler.transform(x_val_split)
    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=100)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_val_pca = pca.transform(x_val_scaled)

    #Grid Search ( + parameters )
    print("Optimizing SVM hyperparameters with Grid Search...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf']
    }
    grid_search = GridSearchCV(
        SVC(class_weight='balanced', random_state=42),
        param_grid,
        cv=3,
        verbose=2
    )
    grid_search.fit(x_train_pca, y_train_split)

    print("\nBest set of parameters found:")
    print(grid_search.best_params_)

    # Evaluate
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(x_val_pca)
    acc = accuracy_score(y_val_split, y_pred)

    print(f"\nOptimized SVM Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val_split, y_pred))

    # Save optimized SVM model, scaler and PCA
    joblib.dump(best_svm, "models/svm_optimized_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(pca, "models/pca.pkl")

    print("\nOptimized SVM model, scaler and PCA saved to 'models/'.")


if __name__ == "__main__":
    optimize_svm_hyperparameters()
