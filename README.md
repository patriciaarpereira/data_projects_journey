# Machine Learning Project

## K-Means Customer Segmentation and Random Forest for Customer Retention Prediction in Brazilian E-Commerce (Olist)

### Project Overview 📘

The goal of this project is to first develop customer segments using clustering techniques aiming to provide valuable insights for optimizing marketing efforts and then build a predictive model to identify customers likely to make repeat purchases. This last model will be based on their purchase history, behavior, and derived customer segments. The scope of the project involves analyzing a public e-commerce dataset from Olist Store, containing information on 100,000 orders placed between 2016 and 2018 across multiple marketplaces in Brazil.


### Technology 🐼

**Python:**
The primary programming language for development, aligned with the concepts learned through Analytics e Data Science Empresarial postgraduation at ISLA and
DataCamp certification "Data Scientist Associate".

- **Libraries**
  - **pandas:** For data manipulation and analysis.
  - **numPy:** For numerical computations.
  - **missingno:** For missing data analysis.
  - **pickle:** Used for saving and loading intermediate Python objects between different scripts by preserving the original data types (unlike CSV).
  - **scipy.stats:** For hypothesis testing during exploratory data analysis (EDA) and feature selection in the machine learning process.
  - **matplotlib.pyplot:** For data visualization.
  - **seaborn:** For enhanced statistical data visualization.
  - **scikit-learn:** For implementing machine learning algorithms (ensemble (RandomForestClassifier), cluster (K-Means), preprocessing (OneHotEncoder, MinMaxScaler), model_selection (train_test_split, KFold, RandomizedSearchCV), metrics (roc_auc_score, confusion_matrix, classification_report, recall_score, accuracy_score, roc_curve, silhouette_score), impute (SimpleImputer), pipeline (Pipeline), compose (ColumnTransformer)).


### Process 🔎

- **CRISP-DM** 
  - The project was developed following the Cross Industry Standard Process for Data Mining (CRISP-DM) model. The scripts and presentation slides were aligned with these phases, as shown below.

- **Machine Learning Methodology**
  - In the Unsupervised Learning scenario with a K-Means algorithm, the data was preprocessed through OneHotEncoder to convert categorical features into a numerical format, and MinMaxScaler to ensure numerical features contribute equally to the distance calculations. The Elbow method and Silhouette score were used to find the best number of clusters ('k'). The final model shows the data's inherent groupings.
  - In the Supervised Learning scenario with a Random Forest algorithm, the data was split into Train (80%) and Test (20%) to ensure unbiased evaluation of the model's ability to generalize. A pipeline, incorporating ColumnTransformer for independent preprocessing of features before cross-validation within Random Search, was crucial to prevent data leakage. Hyperparameter tuning was performed using Random Search with cross-validation, and Recall was chosen as the evaluation metric over accuracy due to an imbalanced target class. The resulting Best Model was then rigorously evaluated on the unseen Test Set to produce the final Random Forest Model.


### Table of contents 📝 

**Datasets Directory**
- "olist_customers_dataset.csv"
- "olist_order_items_dataset.csv"
- "olist_order_payments_dataset"
- "olist_order_reviews_dataset"
- "olist_orders_dataset"
- "olist_products_dataset"
- "olist_sellers_dataset"

**Python Scripts Directory**
- The scripts should be run in the following order:
  - Data Understanding: "P1_Olist_InitialEDA.py"
  - Data Preparation: "P1_Olist_EDA_DataPreparation.py"
  - Modeling and Evaluation: "P1_Olist_Modeling_KMeans.py" and "P1_Olist_Modeling_RandForest.py"

**Presentation**
- A presentation detailing the project findings and methodology, titled "P1_Olist_Presentation," was developed in PPT and provided in PDF format.
  - Business Understanding: "Project" and "Olist and Marketplace Benchmarking" slides.
  - Data Understanding: "Schema, Outliers and Missing Values" and "Initial Insights" slides.
  - Data Preparation: "Insights", "Insights per Retained Customer" and "Features Table" slides.
  - Modeling: "K-Means", "Cluster Analysis (Customer Segmentation)", "Random Forest Methodology" slides.
  - Evaluation: "Random Forest Metrics" slide.
  - Deployment: "Conclusions" and "Business Recommendations" slides.


### How the project came about 💡

Originated as a workgroup assignment for the "Projeto I" course in the "Analytics e Data Science Empresarial" post-graduation at ISLA, was subsequently developed further as a personal initiative.


### The motivation 💥

Developed with the primary goal of showcasing supervised and unsupervised machine learning models regarding my practical skills for entry-level data scientist positions, this project serves as a key element in my professional portfolio.


### Challenges ❓

The project faced a severely imbalanced target class, with only 3% of customers making repeat purchases. To correctly identify retained customers (positive class), recall was the primary evaluation metric. Analysis of the ROC/AUC curve and the precision-recall trade-off, highlighted the need for a cost-benefit analysis regarding false positives. A key business recommendation stemming from this work is to leverage the project's customer segmentation for targeted marketing strategies with the goal to increase customer retention rate.  


### Credits ⭐

Kaggle: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
