# -*- coding: utf-8 -*-
"""
Project - Brazilian E-Commerce Public Dataset by Olist (Retention Prediction)

Modeling - K-Means (Customer Segmentation)

@author: Patrícia Pereira

"""


# --------------------------------------------------------------------
# ### Import Packages ###

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import silhouette_score


# --------------------------------------------------------------------
# ### Load features_analysis_df from Initial Data Prepation py file ###


with open('feature_table_kmeans.pkl', 'rb') as file:
    feature_table = pickle.load(file)


# Set customer_unique_id as index to avoid issues with concatenate
feature_table = feature_table.set_index("customer_unique_id")

# Inspect
feature_table.info()


# Random State
seed = 123



# --------------------------------------------------------------------
# ### Modeling - K-Means (Customer Segmentation) ###


# ----------------------------------------
## Select Relevant Features ##


# Select Features for Analysis
kmeans_features = feature_table[["avg_n_installments",
                                 "recency"]]

kmeans_features.info()



# ----------------------------------------
## Preprocessing ##


# One-hot Enconding for Categorical Variables
cat_var = kmeans_features.select_dtypes("O")
ohe = OneHotEncoder()
cat_var_ohe = ohe.fit_transform(cat_var).toarray()


# Scaler for Numerical Variables
num_var = kmeans_features.select_dtypes("number")
scaler = MinMaxScaler()
num_var_scaler = scaler.fit_transform(num_var)

# Transform do DF
cat_var_ohe_df = pd.DataFrame(
    data = cat_var_ohe,
    columns = ohe.get_feature_names_out(cat_var.columns),
    index = kmeans_features.index)

num_var_scaler_df = pd.DataFrame(
    data = num_var_scaler,
    columns = num_var.columns,
    index = kmeans_features.index)

# Concatenate
kmeans_feat_trasnf = pd.concat([cat_var_ohe_df, num_var_scaler_df], 
                               axis = 1)

# Inspect
kmeans_feat_trasnf.info()



# ----------------------------------------
## Adjust features from One-hot Encoding ##

#kmeans_feat_trasnf = kmeans_feat_trasnf.drop(["pref_pay_type_credit_card",
#                                             "pref_pay_type_voucher",
#                                             "pref_pay_type_boleto"],
#                                             axis = 1)



# ----------------------------------------
## Define the nº of clusters ##


# Elbow Criterion Method
sse = {}

for k in range(1,11):
    kmeans = KMeans(n_clusters = k, random_state = seed)
    kmeans.fit(kmeans_feat_trasnf)
    sse[k] = kmeans.inertia_    # sum of squared distances to closest cluster center

# Plot
sns.pointplot(sse)
plt.title ("Elbow Criterion Method")
plt.xlabel("k")
plt.ylabel("SSE")
plt.show()


# Silhouette Coefficient
silh_score = {}

for k in range(2, 11):
    kmeans = KMeans(n_clusters = k, random_state = seed)
    kmeans.fit(kmeans_feat_trasnf)
    pred = kmeans.predict(kmeans_feat_trasnf)
    score = silhouette_score(kmeans_feat_trasnf, pred)
    silh_score[k] = score

for k, score in silh_score.items():
  print(f"k {k}: {silh_score[k]:.3f}")



# ----------------------------------------
## Define k manually ##

k_final = 4



# ----------------------------------------
## Running K-Means ##

# Run and Fit
kmeans = KMeans(n_clusters = k_final, random_state = seed)
kmeans.fit(kmeans_feat_trasnf)

# Get Labels
cluster_labels = kmeans.labels_    # same result as predict  or fit_predict
cluster_labels_df = pd.DataFrame(cluster_labels, 
                                 columns = ["cluster"],
                                 index = kmeans_features.index)


# Final DF with transformed data
kmeans_feat_trasnf_labels = pd.concat(
    [kmeans_feat_trasnf, cluster_labels_df],
    axis = 1)

# Inspect
kmeans_feat_trasnf_labels.info()



# ----------------------------------------
## Snake Plot ##

# Prepare data
df_melt = pd.melt(kmeans_feat_trasnf_labels,
                     id_vars = "cluster",
                     var_name = "feature",
                     value_name = "value")

# Inspect
df_melt.info()

# Plot
sns.lineplot(data = df_melt,
             x = "feature",
             y = "value",
             hue = "cluster",
             palette = ["orange", "green", "blue", "purple"])
plt.title("Snake Plot", pad = 15)
plt.xticks(["recency", "avg_n_installments"], 
           labels = ["Recency", "Average Nº\nof Installments"])
plt.ylabel("Value")
plt.xlabel("Features")
plt.show()



# ----------------------------------------
## Analysis ##

# DF
feature_table_rf = pd.concat([feature_table, cluster_labels_df],
                             axis = 1)

# Change cluster feature data type
feature_table_rf["cluster"] = feature_table_rf["cluster"].astype("O")

# Inspect
feature_table_rf.info()

# Count
feature_table_rf["cluster"].value_counts(normalize = True)


# --------------------------------------------------------------------
# ### Pickle: feature_table ###


# Transform feature_table to pickle file for modeling

file_path = "feature_table_rf.pkl"
df = feature_table_rf

# Open the file in binary write mode ('wb')
with open(file_path, 'wb') as file:
    # Use pickle.dump() to serialize the DataFrame and write it to the file
    pickle.dump(df, file)

