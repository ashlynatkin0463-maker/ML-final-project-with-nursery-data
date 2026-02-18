# Predicting Nursery School Application Outcomes Using Machine Learning #

### Introduction ###

This is my Machine Learning project that I focused on predicting nursery school application outcomes using a dataset of applicant information. The dataset includes features such as: Parents’ background (parents), Access to a nursery (has_nurs), Application form (form), Number of children (children), Housing situation (housing), Financial status (finance), Social status (social), and Health status (health). Citation :Rajkovic, V. (1989). Nursery [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5P88W.

The target variable is class, which categorizes applications as recommended, priority, or not recommended.

My goal was to build a machine learning model that most accurately classified the applications while also providing interpretability into which features most influencal for nursery admission. In my project I chose to compare a Logistic Regression model and Gradient Boosting Classifier model to determine which of the two had the highest predictive accuracy.

### Data Preprocessing ###

Since most all features are categorical I used One-Hot Encoding to convert the categorical feature into binary indicators. For example, parents with values usual, pretentious, and great_pret they became separate columns. I applied Train-Test Split to split the dataset into 80% training and 20% testing.

Handling Class Imbalance

Some classes, such as very_recom, are underrepresented.

Logistic Regression can handle imbalance with class_weight='balanced'.

Gradient Boosting can capture class relationships inherently via tree splitting.

### Methodology ###

Two models were implemented:

1. Logistic Regression

Preprocessing pipeline:

One-Hot Encoding for categorical features

Logistic Regression classifier with max_iter=1000 and class_weight='balanced'


2. Gradient Boosting Classifier

Tree-based ensemble model capable of capturing non-linear relationships.

Preprocessing pipeline:

One-Hot Encoding for categorical features

Gradient Boosting classifier with tuned hyperparameters (n_estimators, learning_rate, max_depth, subsample)

Evaluated using the same metrics as Logistic Regression.

Feature importance visualizations help interpret which features drive predictions.

### Evaluation: ###

Accuracy- overall how well did it predict

Confusion Matrix - which classes got mixed up 

Classification Report (Precision, Recall, F1-score) - how well was it balanced

5-Fold Cross-Validation - more stable estimate of model performance across multiple different folds

### Results: ###
Class Distribution I Visualized with a bar plot to check the balance. Most classes are evenly represented, except very_recom, which is rare.

Model Performance - 
Logistic Regression	~0.69	Linear model; it handles imbalance in class weights pretty well.
Gradient Boosting	~0.98	Captures complex patterns; interpretable especially feature importance; higher accuracy with categorical data. (better model with this data)

### Conclusion: ###

The ensemble Tree-based model surpasses Logistic Regression with the categorical nursery data because it captured the non-linear relationships and the features of importances. A problem that I ran into was the class imbalance which required me to use a weighted model for the rare case.

Top three features for prediction: Health status (health), Number of children (children), and Parents’ background (parents).
