# Home Credit Default Risk 

In this notebook, I'm going to predict the probability of loan default risk base on partial data from the [Home Credit default risk machine learning competition](https://www.kaggle.com/c/home-credit-default-risk/) in Kaggle. The goal of this competition is to use historical loan application data to predict whether or not an applicant will be able to repay a loan. This is a supervised learning task and classification problem.


## Data

In this notebook, I'm using only the application_train.csv which is 1 file out of 7.
This file is the main table, static data for all applications which one row represents one loan in the data sample.

The description of the columns is in the columns_description.csv , this is a reduced version that describes the application set only.

The data contains 307,511 rows (each row represents a loan) and 122 columns represent information about the client and the loan.

Link to the full data source: https://www.kaggle.com/c/home-credit-default-risk/data


## Metric

In this case, we are dealing with imbalanced data (most of the clients repay their loan) so accuracy matric is not relevant.
So, the metric that we are going to check is the [Receiver Operating Characteristic Area Under the Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) (ROC AUC).

**Reminder**: ROC curve is the true positive rate versus the false positive rate.
The Area Under the Curve (AUC) is the integral of the curve (ROC). This metric provides scores between 0 and 1, higher-score means a better model. A random guessing model will have a ROC AUC score of 0.5.


## Results

The XGBoost provides the best performance from all the algorithms I used (including ANN).
The test ROC AUC score is: **0.7689**
In order to make a comparison, we can look at the best score from this competition which is 0.81 (by using all data files).


## Notebook Content

- Import Data and Libraries
- Exploratory Data Analysis (EDA)
- Data cleaning
- Dealing with Na's
- Feature Engineering
- Data Preparation for Modeling
- Models
- Machine Learning Algorithms
- Hyperparameter Tuning
- Deep Neural Network
- Best Results
- Feature Importances
