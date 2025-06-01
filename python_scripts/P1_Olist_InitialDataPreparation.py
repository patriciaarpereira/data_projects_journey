# -*- coding: utf-8 -*-
"""
Project - Brazilian E-Commerce Public Dataset by Olist (Retention Prediction)

Initial Data Preparation

@author: Patrícia Pereira

"""


# --------------------------------------------------------------------
# ### Import Packages ###

import pickle
import pandas as pd
import numpy as np



# --------------------------------------------------------------------
# ### Load Datasets from Initial EDA py file ###

# As shown in Initial EDA, there is only 1.2 products on average per order and 
# only 1.04 orders on average per client. Dut to the lack of diversity, we will
# only consider the number of products and sellers for further analysis, but 
# will not consider additional details about products and sellers datasets.

with open('customers.pkl', 'rb') as file:
    customers = pickle.load(file)

with open('orders.pkl', 'rb') as file:
    orders = pickle.load(file)

with open('order_items.pkl', 'rb') as file:
    order_items = pickle.load(file)

with open('reviews.pkl', 'rb') as file:
    reviews = pickle.load(file)

with open('payments.pkl', 'rb') as file:
    payments = pickle.load(file)


# Inspect
customers.info()
orders.info()
order_items.info()
reviews.info()
payments.info()



# --------------------------------------------------------------------
# ### Create the Target Variable (Retention) ###


# ----------------------------------------
## Delivered Orders ##


# As mentioned on the Initial EDA py file, the delivered orders represent 
# around 97% of total orders. In addition, there are missing dates regarding 
# different order status that could be relevant for retention prediction.
# Therefore, we will only consider delivered orders

ord_del = orders[orders["order_status"] == "delivered"]
ord_del = ord_del.drop("order_status", axis = 1)

# Inspect
ord_del.info()



# ----------------------------------------
## Date Split ##


# Split date
split_date = "2017-05-31 23:59:59"


# New DataFrame
ret_target = customers[
    ["customer_unique_id", "customer_id"]].merge(ord_del[
        ["customer_id",
         "order_purchase_timestamp"]],
        how = "left",
        on = "customer_id")

        
# Split the customers 
# reset_index to ensure the index are correct when creating the DF
cust_before_split = ret_target.loc[
    ret_target["order_purchase_timestamp"] <= split_date,
    "customer_unique_id"].drop_duplicates().reset_index(drop = True)

cust_after_split = ret_target.loc[
    ret_target["order_purchase_timestamp"] > split_date, 
    "customer_unique_id"].drop_duplicates()


# Target Variable
# 1 for customers that bought again after the split date
# 0 for customers that bought before but not after the split date
target = np.where(cust_before_split.isin(cust_after_split), 1, 0)



# ----------------------------------------
## Retention Target Dataframe ##


# Final DataFrame
retention_target = pd.DataFrame({
    "customer_unique_id": cust_before_split,
    "target": target})

# Inspect
retention_target.info()
retention_target.head()

# Imbalanced Class with only 3% of retention
retention_target["target"].value_counts(normalize = True)



# --------------------------------------------------------------------
# ### Data Preparation for EDA ###


# ----------------------------------------
## Data Prepation - data leakage / customer_unique_id / order_id ##


# Define orders before split date to avoid data leakage in the analysis
ord_del_bef_split = ord_del[ord_del["order_purchase_timestamp"] <= split_date]

# Inspect
ord_del_bef_split.info()


# Create a DF with only customer_unique_id and order_id to be used further on
# joins
custunid_orderid = customers[[
    "customer_unique_id", "customer_id"]].merge(
        ord_del_bef_split[["customer_id", "order_id"]],
        how = "right",
        on = "customer_id")

custunid_orderid = custunid_orderid.drop("customer_id", axis = 1)
        
# Inspect
custunid_orderid.info()
custunid_orderid["customer_unique_id"].nunique()



# ----------------------------------------
## Customers datasets ##


# Feature Engineering
cust_features = customers.groupby(
    "customer_unique_id")["customer_state"].first().reset_index()

# Inspect
cust_features.head()


# Join to Final Features Table
features_analysis_df = retention_target.merge(cust_features,
                                              how = "left",
                                              on = "customer_unique_id")

# Inspect
features_analysis_df.head()
features_analysis_df.info()



# ----------------------------------------
## Orders datasets ##


# Feature Engineering
# Create new waiting time featurese in number of day
ord_del_bef_split["diff_days_del_purch"] = (ord_del_bef_split[
    "order_delivered_customer_date"] - ord_del_bef_split[
        "order_purchase_timestamp"]).dt.total_seconds() / (60*60*24)

ord_del_bef_split["diff_days_estdel_del"] = (ord_del_bef_split[
    "order_estimated_delivery_date"] - ord_del_bef_split[
        "order_delivered_customer_date"]).dt.total_seconds() / (60*60*24)

        
# Join to customer_unique_id
ord_del_bef_split_cust = custunid_orderid.merge(ord_del_bef_split,
                                              how = "left",
                                              on = "order_id")

        
# Features by customer_unique_id
ord_features = ord_del_bef_split_cust.groupby("customer_unique_id").agg(
    recency = ("order_purchase_timestamp", lambda x: pd.to_datetime(split_date) - x.max()),
    frequency = ("order_id", "count"),
    tenure = ("order_purchase_timestamp", lambda x: pd.to_datetime(split_date) - x.min()),
    avg_diff_days_del_purch = ("diff_days_del_purch", "mean"),
    avg_diff_days_estdel_del = ("diff_days_estdel_del", "mean")
).reset_index()

# Change data types from timedelta to nº of days as float
ord_features["recency"] = ord_features[
    "recency"].dt.total_seconds() / (60*60*24)

ord_features["tenure"] = ord_features[
    "tenure"].dt.total_seconds() / (60*60*24)


# Join to Final Features Table
features_analysis_df = features_analysis_df.merge(ord_features,
                                              how = "left",
                                              on = "customer_unique_id")

# Inspect
features_analysis_df.head()
features_analysis_df.info()

# Drop the only 1 missing value
features_analysis_df = features_analysis_df.dropna(subset = "avg_diff_days_del_purch")



# ----------------------------------------
## Order Items datasets ##


# Feature Engineering
# Group some features by order_id first in order to avoid errors on 
# customer_unique_id aggregation

# Features by order_id
ord_ord_it_features = order_items.groupby("order_id").agg(
    quant_prod = ("order_item_id", "max"),
    tt_freight_value = ("freight_value", "sum"),
    tt_amount = ("price", "sum"))

ord_ord_it_features["freight_rate"] = ord_ord_it_features[
    "tt_freight_value"] / ord_ord_it_features["tt_amount"]


# Join with custunid_orderid DF
ord_item_cust = custunid_orderid.merge(order_items,
                                            how = "left",
                                            on = "order_id")

ord_it_ord_cust = custunid_orderid.merge(ord_ord_it_features,
                                            how = "left",
                                            on = "order_id")

# Inspect
ord_item_cust.info()
ord_it_ord_cust.info()


# Features by customer_unique_id
ord_item_features = ord_item_cust.groupby("customer_unique_id").agg(
    tt_unique_sellers = ("seller_id", "nunique"),
    monetary_value = ("price", "sum"),
    tt_unique_products = ("product_id", "nunique")).reset_index()

ord_item_ord_features =  ord_it_ord_cust.groupby("customer_unique_id").agg(
    avg_quant_prod = ("quant_prod", "mean"),
    avg_freight_value = ("tt_freight_value", "mean"),
    avg_freight_rate = ("freight_rate", "mean")).reset_index()


# Join to Final Features Table
features_analysis_df = features_analysis_df.merge(ord_item_features,
                                              how = "left",
                                              on = "customer_unique_id")

features_analysis_df = features_analysis_df.merge(ord_item_ord_features,
                                              how = "left",
                                              on = "customer_unique_id")

# Inspect
features_analysis_df.head()
features_analysis_df.info()



# ----------------------------------------
## Payments datasets ##


# Feature Engineering
# Group some features by order_id first in order to avoid errors on 
# customer_unique_id aggregation

# Features by order_id
# By applying the mode, we have the issue that there can be two modes. In
# these situations, we assumed iloc[0] which is "credit_card" as it is the 
# most frequent payment type.
pay_order_features = payments.groupby("order_id").agg(
    n_installments = ("payment_installments", "sum")).reset_index()


# Join with custunid_orderid DF
pay_order_cust = custunid_orderid.merge(pay_order_features,
                                            how = "left",
                                            on = "order_id")

pay_cust = custunid_orderid.merge(payments,
                                  how = "left",
                                  on = "order_id")

# Inspect
pay_cust.info()
pay_order_cust.info()


# Features by customer_unique_id
pay_cust_feat = pay_cust.groupby("customer_unique_id").agg(
    pref_pay_type = ("payment_type", 
                     lambda x: x.mode().iloc[0] 
                     if not x.mode().empty 
                     else None)).reset_index()

pay_features = pay_order_cust.groupby("customer_unique_id").agg(
    avg_n_installments = ("n_installments", "mean")).reset_index()


# Join to Final Features Table
features_analysis_df = features_analysis_df.merge(pay_features,
                                              how = "left",
                                              on = "customer_unique_id")

features_analysis_df = features_analysis_df.merge(pay_cust_feat,
                                              how = "left",
                                              on = "customer_unique_id")

# Inspect
features_analysis_df.head()
features_analysis_df.info()

# Drop the only missing payment type and n_installments
features_analysis_df = features_analysis_df.dropna(
    subset = "avg_n_installments")



# ----------------------------------------
## Reviews datasets ##


# Feature Engineering
# Group some features by order_id first in order to avoid errors on 
# customer_unique_id aggregation

# Feature by order_id
rev_ord_feature = reviews.groupby("order_id").agg(
    avg_review_score = ("review_score", "mean")).reset_index()


# Join with custunid_orderid DF
rev_ord_cust = custunid_orderid.merge(rev_ord_feature,
                                      how = "left",
                                      on = "order_id")


# Feature by customer_unique_id
rev_feature = rev_ord_cust.groupby("customer_unique_id").agg(
    avg_review_score = ("avg_review_score", "mean")).reset_index()


# Join to Final Features Table
features_analysis_df = features_analysis_df.merge(rev_feature,
                                              how = "left",
                                              on = "customer_unique_id")

# Inspect
features_analysis_df.head()
features_analysis_df.info()



# --------------------------------------------------------------------
# ### Pickle: features_analysis_df ###


# Transform features_analysis_df to pickle file for EDA analysis and final
# selection of relevant features 

file_path = "features_analysis_df.pkl"
df = features_analysis_df

# Open the file in binary write mode ('wb')
with open(file_path, 'wb') as file:
    # Use pickle.dump() to serialize the DataFrame and write it to the file
    pickle.dump(df, file)
