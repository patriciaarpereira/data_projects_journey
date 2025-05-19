# -*- coding: utf-8 -*-
"""
Project - Brazilian E-Commerce Public Dataset by Olist (Retention Prediction)

Modeling - Random Forest (Retention Prediction)

@author: Patrícia Pereira

"""


# --------------------------------------------------------------------
# ### Import Packages ###

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, \
    classification_report, recall_score, accuracy_score, roc_curve
from sklearn.model_selection import KFold, RandomizedSearchCV



# --------------------------------------------------------------------
# ### Load features_analysis_df from Initial Data Prepation py file ###


with open('feature_table_rf.pkl', 'rb') as file:
    feature_table_rf = pickle.load(file)

# Inspect
# The index is the customer_unique_id
feature_table_rf.info()


# Random State
seed = 123

 

# --------------------------------------------------------------------
# ### Modeling - Random Forest (Retention Prediction) ###


# ----------------------------------------
## Define Variables ##

X = feature_table_rf[["avg_diff_days_del_purch",
                      "avg_review_score",
                      "cluster"]]

y = feature_table_rf["target"]



# ----------------------------------------
## Train / Test Split ##

# Split the dataset into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    stratify = y,
                                                    random_state = seed)


# ----------------------------------------
## Preprocessing ##

# One-hot Enconding for Categorical Variables
X_cat_var = X.select_dtypes("O")
ohe = OneHotEncoder()


# Imputation for missing values
# As avg_review_score is skewed, selecting the median (instead of mean)
X_num_var = X[["avg_review_score"]]
imp_num = SimpleImputer(strategy = "median")


# This step is needed because there are different kind of imputations for
# categorical and numerical features
# remainder = "passthrough otherwise it will not get the other features
preprocessor = ColumnTransformer(
    transformers = [
        ("cat", ohe, X_cat_var.columns),
        ("num", imp_num, X_num_var.columns)],
    remainder = "passthrough")



# ----------------------------------------
## Run Random Forest  ##

rf_model = RandomForestClassifier(random_state = seed)



# ----------------------------------------
## Pipeline ##

# Pipeline
# The step "column_remover is for removing the clusters that are not needed,
# according to feature importance from the first version.
pipeline = Pipeline(
    steps = [("preprocessor", preprocessor),
             ("column_remover", FunctionTransformer(
                 lambda array: pd.DataFrame(array).drop(
                     columns = [1, 2]))),
             ("rf_model", rf_model)])

# Fit
pipeline.fit(X_train, y_train)

# Check names to remove the correct index and keep only relevant clusters
print(pipeline.named_steps['preprocessor'].get_feature_names_out())

# Predict
y_pred = pipeline.predict(X_test)



# ----------------------------------------
## Evaluation (Baseline) ##


# Confusion Matrix
print(confusion_matrix(y_test, y_pred), "\n")


# Classification Report 
print(classification_report(y_test, y_pred), "\n")


# ROC Curve / AUC
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred):.3f}")



# ----------------------------------------
## CV / RandomnizedSearch ##


# Shuffle and split the data
kf = KFold(n_splits = 10, shuffle = True, random_state = seed)


# Define the grid of hyperparameters
params_rf = {
    'rf_model__n_estimators': [200, 250, 300],
    "rf_model__max_depth": [5, 6, 8],
    "rf_model__min_samples_leaf": [0.22, 0.25, 0.27],
    "rf_model__max_features": [1, 2, 3],    # not relevant as they are only few
    "rf_model__class_weight": ["balanced", "balanced_subsample", None]}


cv_rf = RandomizedSearchCV(estimator = pipeline,
                           param_distributions = params_rf,
                           scoring = "recall",
                           cv = kf,
                           n_iter = 30,
                           random_state = seed)


# Fit cv_rf to the training set
cv_rf.fit(X_train, y_train)



# ----------------------------------------
## Evaluation (CV) ##


# Extract best hyperparameters from cv_rf
best_hyperparams = cv_rf.best_params_
print("Best hyerparameters:\n", best_hyperparams, "\n")


# Extract best CV score (recall) from cv_rf
# Mean cross-validated score of the best_estimator
best_CV_score = cv_rf.best_score_
print(f"Best CV score (recall): {best_CV_score:.3f}")


# Extract best model from cv_rf
best_model = cv_rf.best_estimator_



# ----------------------------------------
## Evaluation (Test Set) ##


# Generalization Error
# Predict of training set
y_pred_train = best_model.predict(X_train)

# Predict of test set
y_pred_test = best_model.predict(X_test)

# CV Recall
print(f"CV Recall: {best_CV_score:.3f}")

# Training set Recall
print(f"Train Recall: {recall_score(y_train, y_pred_train):.3f}")

# Test set Recall
print(f"Test Recall: {recall_score(y_test, y_pred_test):.3f}")


# Accuracy
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.3f}", "\n")


# Confusion Matrix
print(confusion_matrix(y_test, y_pred_test), "\n")


sns.heatmap(confusion_matrix(y_test, y_pred_test),
            annot=True,
            fmt='d',
            cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix', pad = 15)
plt.show()


# Classification Report
print(classification_report(y_test, y_pred_test))


# ROC Curve / AUC

y_pred_probs_test = best_model.predict_proba(X_test)[:,1]

# Plot
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs_test)
plt.plot([0, 1], [0, 1], 'k--', color = "cyan")
plt.plot(fpr, tpr, color = "blue")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest ROC Curve", pad = 15)
plt.show()

# ROC AUC area
print("\n", f"ROC AUC Area: {roc_auc_score(y_test, y_pred_probs_test):.3f}")



# ----------------------------------------
## Feature Importance ##


# Get feature importances from the RandomForest model from the pipeline
feature_importances = best_model.named_steps[
    "rf_model"].feature_importances_

# Get the feature names from the preprocessor
feature_names = best_model.named_steps[
    "preprocessor"].get_feature_names_out().tolist()

print(feature_names)

# As clusters 1 and 2 were removed due to lack of feature importance from 
# the first version, we need to adjust the new names
# New feature names
new_names = ["Cluster 0", 
             "Cluster 3", 
             "Average Review\nScore", 
             "Average Delivery\nTime (Days)"]

# Create a pd.Series for feature importance
feature_importance_rf = pd.Series(
    feature_importances, index = new_names).sort_values(
        ascending = False)

# Plot
sns.barplot(y = feature_importance_rf.index,
            x = feature_importance_rf.values)
plt.title("Feature Importance", pad = 15)
plt.ylabel("")
plt.show()
