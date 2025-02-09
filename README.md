Titanic Survival Prediction


This project analyzes Titanic passenger data to predict the likelihood of survival using machine learning, specifically the Random Forest Classifier. The project includes various data preprocessing steps, visualizations, and model evaluation techniques.


Description
The Titanic dataset contains information about passengers aboard the Titanic, including details like passenger class, age, sex, fare, and whether they survived. The goal is to predict the likelihood of survival based on these features.

This code includes:
•	Data cleaning (handling missing values, duplicates, and irrelevant columns).
•	Exploratory Data Analysis (EDA) with visualizations to understand the distribution of data and correlations.
•	Building a predictive model using a Random Forest Classifier.
•	Model evaluation with accuracy, classification report, confusion matrix, and ROC curve.


Prerequisites
To run this project, you’ll need the following libraries installed:
•	pandas
•	numpy
•	matplotlib
•	seaborn
•	scikit-learn
•	openpyxl (for reading Excel files)


You can install these using pip:
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl

Installation
1.	Clone this repository to your local machine:
git clone https://github.com/yourusername/titanic-survival-prediction.git
2.	Navigate to the project directory:
cd titanic-survival-prediction
3.	Make sure you have installed all the dependencies as mentioned in the prerequisites.
4.	Download the Titanic dataset (titanic3.xls) and place it in the same directory as the script.


Usage
1.	Run the script using Python 3.x:
python titanic_survival_prediction.py
2.	The script will:
o	Load and preprocess the dataset.
o	Perform exploratory data analysis (EDA) including visualizations.
o	Train a Random Forest Classifier model.
o	Evaluate the model and display performance metrics (accuracy, confusion matrix, ROC curve).


Data Preprocessing
•	Missing values are handled by filling the age column with its mean value and dropping rows with missing values in the embarked and fare columns.
•	The cabin column is removed due to too many missing values.
•	The sex column is encoded into numerical values (0 for male, 1 for female).
•	The survived column is converted from categorical values ('Survived' and 'Did not survive') to numeric values (1 and 0).


Visualizations
The script generates the following visualizations:
•	A histogram and box plot for the age distribution.
•	A pie chart showing the survival rate.
•	A stacked bar chart for survival based on sex.
•	A heatmap for correlation between numerical features.
•	A pie chart for class distribution (pclass).
•	A ROC curve and confusion matrix for model evaluation.


Model Evaluation
The model's performance is evaluated based on:
•	Accuracy: The proportion of correctly classified instances.
•	Classification Report: Precision, recall, and F1-score for each class.
•	Confusion Matrix: A matrix that summarizes the classification results.
•	ROC Curve: A curve showing the trade-off between true positive rate and false positive rate.
•	AUC-ROC Score: The area under the ROC curve, indicating the model’s ability to discriminate between classes.


Example Output
The script will output:
•	The accuracy of the Random Forest Classifier model.
•	The classification report (precision, recall, F1-score).
•	The confusion matrix (visualized as a heatmap).
•	The ROC curve and AUC-ROC score.


Example output for accuracy:
Accuracy: 0.82
Classification Report:
              precision    recall  f1-score   support

   Did not survive       0.83      0.83      0.83        113
        Survived       0.81      0.80      0.81         87

    accuracy                           0.82        200
   macro avg       0.82      0.82      0.82        200
weighted avg       0.82      0.82      0.82        200


Contributing
Feel free to fork this project and submit pull requests. Contributions are welcome!
1.	Fork the repository.
2.	Create a new branch (git checkout -b feature-branch).
3.	Make your changes and commit them (git commit -am 'Add new feature').
4.	Push to the branch (git push origin feature-branch).
5.	Open a Pull Request.
