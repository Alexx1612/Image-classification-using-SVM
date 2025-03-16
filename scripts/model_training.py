import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_save_svm():
    """
    Complete SVM training pipeline with optimized parameters.
    """
    # Ensure required directories exist (create if needed)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("Load training data...")
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")["Label"]

    # Generate class distribution plot
    plot_class_distribution("data/y_train.csv", "reports")

    # Spliting data
    print("Creating train/validation split (80/20)...  Applying feature standardization...  Applying PCA for dimensionality reduction...")
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Feature standardization
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_split)
    x_val_scaled = scaler.transform(x_val_split)

    # Dimensionality reduction
    pca = PCA(n_components=100)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_val_pca = pca.transform(x_val_scaled)

    print("Training SVM with optimized hyperparameters...  Evaluating model performance...")
    model = SVC(C=10, gamma=0.001, kernel='rbf', class_weight='balanced', random_state=42)
    model.fit(x_train_pca, y_train_split)

    # Evaluating model performance
    y_pred = model.predict(x_val_pca)
    acc = accuracy_score(y_val_split, y_pred)
    print(f"SVM Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_val_split, y_pred))

    # Save Confusion Matrix
    plot_confusion_matrix(y_val_split, y_pred, "reports")

    # Save model, scaler and PCA
    joblib.dump(model, "models/svm_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(pca, "models/pca.pkl")
    print("SVM model, scaler and PCA saved to 'models/'.")

    return y_val_split, y_pred

def plot_class_distribution(y_train_path, report_dir):
    """
    Generate and save class distribution visualization (as bar chart).
    """
    y_train = pd.read_csv(y_train_path)["Label"]
    plt.figure(figsize=(8, 6))
    y_train.value_counts().sort_index().plot(kind="bar", color="skyblue")
    plt.title("Training Data Class Distribution")
    plt.xlabel("Class labels (0-9)")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(report_dir, "class_distribution.png"))
    plt.close()
    print("Class distribution saved to 'reports/class_distribution.png'.")


def plot_confusion_matrix(y_true, y_pred, report_dir):
    """
    Generate and save confusion matrix visualization (as heatmap).
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(report_dir, "confusion_matrix.png"))
    plt.close()
    print("Confusion Matrix saved to 'reports/confusion_matrix.png'.")


if __name__ == "__main__":
    train_and_save_svm()
