import pandas as pd
import joblib
import os

def load_model_and_predict(test_path, model_path, scaler_path, pca_path, output_path):
    # model, scaler, PCA  +  test data
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    print("Model, scaler, and PCA loaded successfully!")

    X_test = pd.read_csv(test_path)
    print("Test data loaded.")
    X_test_scaled = scaler.transform(X_test) # scaled using pre-fitted scaler
    X_test_pca = pca.transform(X_test_scaled) # reducing to principal components

    # predictions
    predictions = model.predict(X_test_pca)
    print("Predictions generated for test data.")

    # Save predictions in .csv file
    submission = pd.DataFrame({"Id": range(len(predictions)), "Label": predictions})
    submission.to_csv(output_path, index=False)
    print(f" Predictions saved to: {output_path}")


if __name__ == "__main__":
    test_path = "data/X_test.csv"
    model_path = "models/svm_model.pkl"
    scaler_path = "models/scaler.pkl"
    pca_path = "models/pca.pkl"
    output_path = "data/submission_svm.csv"

    load_model_and_predict(test_path, model_path, scaler_path, pca_path, output_path)
