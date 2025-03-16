import numpy as np
import pandas as pd


def npz_to_csv(train_path, test_path, output_dir):
    """
    Convert .npz files to .csv format
    """
    # Loads training data (train.npz)
    train_data = np.load(train_path)
    print(f"Keys in train.npz: {list(train_data.keys())}")

    # Extract training components
    X_train = train_data['x_train']  # Training features matrix
    y_train = train_data['y_train']  # Training labels vector

    # Save training data (train.csv)
    pd.DataFrame(X_train).to_csv(f"{output_dir}/X_train.csv", index=False)
    pd.DataFrame(y_train, columns=["Label"]).to_csv(f"{output_dir}/y_train.csv", index=False)
    print("train.npz converted to X_train.csv and y_train.csv!")

    # Load test data (test.npz)
    test_data = np.load(test_path)
    print(f"Keys in test.npz: {list(test_data.keys())}")

    # Extract test features
    X_test = test_data['x_test']

    # Save test data (test.csv)
    pd.DataFrame(X_test).to_csv(f"{output_dir}/X_test.csv", index=False)
    print("test.npz converted to X_test.csv!")


if __name__ == "__main__":
    train_path = "data/train.npz"
    test_path = "data/test.npz"
    output_dir = "data"

    # Execute conversion
    npz_to_csv(train_path, test_path, output_dir)
