# Credit Card Fraud Detection using Machine Learning

![image](https://github.com/Abhaykumar04/Credit-Card-Fraud-Detection/assets/112232080/0e61e09f-5a85-4212-b4ba-5c60362ec009)


In this project, titled "Credit Card Fraud Detection using Machine Learning," I have undertaken the task of developing a robust system to identify fraudulent credit card transactions. By leveraging various machine learning algorithms, I aim to detect fraudulent activities and minimize potential financial losses for credit card companies and their customers. The models implemented for this purpose include Logistic Regression, Support Vector Machine, Naive Bayes, K-Nearest Neighbors, Random Forest, AdaBoost, and XGBoost.

### Data Definition

The data used for this project consists of credit card transactions, where each transaction record contains various features such as transaction amount, timestamp, and anonymized customer information. The dataset is labeled, with each transaction marked as either fraudulent or non-fraudulent. By utilizing this labeled data, we can train supervised machine learning models to learn patterns and make predictions based on the provided features.

Checkout detaset from : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Logistic Regression

Logistic Regression is a widely-used classification algorithm that models the relationship between the input features and the probability of an event occurring. In this case, we employ logistic regression to classify credit card transactions as either fraudulent or non-fraudulent based on the provided features. By fitting a logistic function to the training data, the model can predict the probability of fraud for each transaction and assign a binary label based on a chosen threshold.

### Support Vector Machine (SVM)

Support Vector Machine is a powerful algorithm for both classification and regression tasks. SVM constructs a hyperplane or set of hyperplanes to separate different classes in the feature space. In credit card fraud detection, SVM can learn to discriminate between genuine and fraudulent transactions by finding the best possible decision boundary. By utilizing different kernels and hyperparameter tuning, we can optimize the SVM model's performance.

### Naive Bayes

Naive Bayes is a probabilistic classifier based on Bayes' theorem and the assumption of independence between features. It is known for its simplicity and computational efficiency. In credit card fraud detection, Naive Bayes calculates the probability of a transaction being fraudulent given its features. By selecting the class with the highest probability, we can classify transactions as fraudulent or non-fraudulent. Despite its naive assumption, Naive Bayes often performs well in real-world scenarios.

### K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a non-parametric classification algorithm that makes predictions based on the k closest neighbors in the feature space. For credit card fraud detection, KNN can identify similarities between transactions and determine the majority class within its k nearest neighbors. By adjusting the number of neighbors and selecting an appropriate distance metric, we can optimize the KNN model's accuracy.

### Random Forest

Random Forest is an ensemble algorithm that combines multiple decision trees to make predictions. Each decision tree is trained on a random subset of features and samples from the dataset. In credit card fraud detection, Random Forest can capture complex relationships between features and effectively handle imbalanced datasets. By aggregating the predictions of individual trees, the model provides a robust and accurate fraud detection mechanism.

### AdaBoost

AdaBoost, short for Adaptive Boosting, is an ensemble method that iteratively combines weak classifiers to create a strong classifier. Each weak classifier is trained on a subset of the data, with higher weights assigned to misclassified samples in the previous iteration. In credit card fraud detection, AdaBoost can focus on difficult-to-classify instances, improving the model's overall performance.

### XGBoost

XGBoost is an optimized implementation of gradient boosting, known for its exceptional performance in various machine learning tasks. Similar to AdaBoost, XGBoost combines weak prediction models to create a robust and accurate classifier. It leverages gradient optimization techniques and regularization to enhance the model's performance and prevent overfitting. XGBoost often achieves state-of-the

-art results and is a popular choice for credit card fraud detection.

### Initial Metrics

To evaluate the performance of the implemented models, we will initially consider the following metrics:

1. Accuracy: It measures the overall correctness of the model's predictions and provides an initial assessment of the model's performance.

2. Precision: Precision quantifies the proportion of correctly classified fraudulent transactions among all the transactions classified as fraudulent. It helps us understand the model's ability to correctly identify actual fraud cases.

3. F1-Score: The F1-Score is the harmonic mean of precision and recall. It provides a balanced measure of the model's performance by considering both precision and recall.

### Main Metrics

In addition to the initial metrics, we will focus on the following metrics:

1. Recall: Recall, also known as sensitivity or true positive rate, measures the proportion of actual fraudulent transactions that the model correctly identifies. It is crucial to minimize false negatives and identify as many fraud cases as possible.

2. AUC/RUC Curve: The Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC) provide insights into the model's trade-off between true positive rate (recall) and false positive rate. By analyzing the ROC curve and calculating the AUC, we can determine the model's performance across different classification thresholds and make informed decisions regarding model selection and threshold optimization.

By considering these metrics, we aim to build an effective credit card fraud detection system that minimizes financial losses and ensures a secure environment for credit card transactions.
