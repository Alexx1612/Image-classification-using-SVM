# Image-classification-using-SVM

This report details our solution for the TIA 2024 Kaggle Competition, a multi-class classification task focused on anomaly detection. The goal was to develop a machine learning model that achieves high accuracy without using neural networks. Our approach utilized a Support Vector Machine (SVM) with hyperparameter optimization, preprocessing techniques, and a clear workflow to achieve 90.17% accuracy on the validation set and secure a top-10 position in the competition.

LINK : https://www.kaggle.com/competitions/tia-2024

🚀 How It Works
1. Data Preparation
- We started by cleaning and formatting the data:
  - Step 1 : Converted .npz files to CSV using convert_npz_to_csv.py. 
  - Step 2 : Standardized the data so all features have the same scale. 

2. Feature Engineering
- To simplify the data and speed up training:
  - PCA (Principal Component Analysis) : Reduced features to 100 key components while keeping most information.

3. Model Training
- Algorithm : SVM with RBF kernel (great for complex classification).

- Hyperparameters :
  - C=10: Controls how strict the model is about errors.
  - gamma=0.001: Determines how "wide" the model’s decision boundaries are.

 - Tuning : Found the best parameters using Grid Search (like a smart trial-and-error process). 

4. Evaluation
- Accuracy : 90.17% on validation data. 
- Confusion Matrix : Showed where the model often confused similar classes (e.g., "Shirt" vs. "T-shirt").

5. Final Predictions
- The trained model was used to predict the test dataset.
- Saved results to submission_svm.csv for Kaggle submission. 

📚 Tools & Technologies
- These are the tools we used:

  - Python : For coding the entire project. 
  - scikit-learn : To train the SVM model and preprocess data. 
  - Pandas/Numpy : For handling and analyzing data. 
  - Matplotlib/Seaborn : To create graphs like confusion matrices. 

📊 Key Results
 - Top-10 Finish : We ranked in the top 10 out of all teams! 🥇
 - Accuracy : 90.17% on validation data.
 - Visualizations :
   - Class Distribution : Shows data is balanced (equal samples for each class). 
   - Confusion Matrix : Highlights common mistakes (e.g., "Shirt" vs. "T-shirt"). 
