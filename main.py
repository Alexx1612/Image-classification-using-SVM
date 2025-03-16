from scripts.model_training import train_and_save_svm
from scripts.generate_reports import plot_class_distribution, plot_confusion_matrix
from scripts.predict_with_svm import load_model_and_predict
import os

def main():
    # Ensure required directories exist (create if needed)
    os.makedirs("reports", exist_ok=True)

    # STEP 1
    print("STEP 1: Training the SVM model...")
    y_val, y_pred = train_and_save_svm()

    # STEP 2
    print("\nSTEP 2: Generating diagnostic reports...")
    plot_class_distribution("data/y_train.csv", "reports")      # Distribution in training data
    plot_confusion_matrix(y_val, y_pred, "reports")                       # Evaluate model performance

    # STEP 3
    print("\nSTEP 3: Final predictions for X_test...")
    load_model_and_predict(
        test_path="data/X_test.csv",        # Raw test dataset file
        model_path="models/svm_model.pkl",  # Trained SVM classification model
        scaler_path="models/scaler.pkl",    # Fitted feature scaler
        pca_path="models/pca.pkl",          # Dimensionality reduction config
        output_path="data/submission_svm.csv"# Output file
    )

    print("\nFinal submission file at:")
    print(" => data/submission_svm.csv")
if __name__ == "__main__":
    main()
