# -*- coding: utf-8 -*-
"""
Project - Brazilian E-Commerce Public Dataset by Olist (Retention Prediction)

Initial EDA

@author: Patrícia Pereira

"""


# --------------------------------------------------------------------
# ### Import Packages ###

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import pickle
import numpy as np
from scipy.stats import mannwhitneyu



# --------------------------------------------------------------------
# ### Define visualization style ###

sns.set(font_scale = 1.4)
sns.set_style("ticks")
sns.set_palette("winter")
pad = 15

blue_color = "#0000FF"
insight_color = "#00FFFF"



# --------------------------------------------------------------------
# ### Customers Dataset ###

customers = pd.read_csv("olist_customers_dataset.csv")



# ----------------------------------------
## Dataset Inspection ##

customers.info()
customers.iloc[:,2:].head()    # the first two columns are IDs
customers.describe()


# ----------------------------------------
## Data Type Constraints ##

# Zip Code as str
customers["customer_zip_code_prefix"] = customers[
    "customer_zip_code_prefix"].astype("str")



# ----------------------------------------
## Uniqueness Constraints ##

# Checking for duplicates in the dataset
customers.duplicated().any()

# Checking for duplicates in customer_id
customers["customer_id"].duplicated().any()

# Checking for duplicates in customer_unique_id
customers["customer_unique_id"].duplicated().any()
customers["customer_unique_id"].nunique()

# Average nº of orders per client
customers["customer_id"].count() / customers[
    "customer_unique_id"].nunique() 

        

# ----------------------------------------
## Categorical Variables ##

# Identify possible category features
customers.nunique()

# Nº of clients per State
n_cli_state = (customers[
    "customer_state"].value_counts(
        normalize = True) * 100).head(10)

# Visualization
custom_palette_state = [insight_color] + [blue_color] * (len(n_cli_state) - 1)

sns.catplot(
    data = n_cli_state,
    kind = "bar",
    palette = custom_palette_state)
plt.title("Proportion of Clients per State", pad = pad)
plt.xlabel("Top 10 States")
plt.ylabel("Proportion (%)")
plt.show()



# ----------------------------------------
## Completeness ##

customers.isna().sum()


# ----------------------------------------
## Notes ##

# - customer_unique_id has duplicates because customer_id is the key to the 
# orders dataset. Each order has a unique customer_id.
# - More than 40% of clients leave in SP
# - Each client made 1.04 orders on average
# 



# ----------------------------------------
## Pickle ##

file_path = "customers.pkl"
df = customers

# Open the file in binary write mode ('wb')
with open(file_path, 'wb') as file:
    # Use pickle.dump() to serialize the DataFrame and write it to the file
    pickle.dump(df, file)



# --------------------------------------------------------------------
# ### Orders Dataset ###

orders = pd.read_csv("olist_orders_dataset.csv")



# ----------------------------------------
## Dataset Inspection ##

orders.info()
orders.iloc[:,2:].head()    # the first two columns are IDs
orders.iloc[:,2:].describe()



# ----------------------------------------
## Data Type Constraints ##

# Dates
ord_date_cols = orders.iloc[:,3:8].columns
orders[ord_date_cols] = orders[ord_date_cols].apply(pd.to_datetime)

# Check
orders.info()
orders[ord_date_cols].head()



# ----------------------------------------
## Uniqueness Constraints ##

# Checking for duplicates in the dataset
orders.duplicated().any()

# Checking for duplicates in id
orders["customer_id"].duplicated().any()

# Checking for duplicates in order_id
orders["order_id"].duplicated().any()
orders["customer_id"].duplicated().any()



# ----------------------------------------
## Categorical Variables ##

# Identify possible category features
orders.nunique()

# Nº of clients per State
n_ord_status = orders["order_status"].value_counts(normalize = True)

n_ord_status_agg = (orders["order_status"].map({
    "delivered": "Delivered"}).fillna(
        "Other").value_counts(normalize = True) * 100).reset_index(name = "prop")

# Visualization
custom_palette_ord = [insight_color] + [blue_color] 

sns.catplot(data = n_ord_status_agg,
            x = "prop",
            y = "order_status",
            kind = "bar",
            palette = custom_palette_ord)
plt.title("Proportion of Orders", pad = pad)
plt.xlabel("Proportion (%)")
plt.ylabel("Status")
plt.show()



# ----------------------------------------
## Completeness ##

orders.isna().sum()

# Check missing dates
msno.matrix(orders[
    ["order_status", "order_delivered_customer_date"]].sort_values(
        by = "order_status", ascending = True))
plt.show()



# ----------------------------------------
## Analyze ##

# Analyze orders over time
orders_date = orders.resample("D", on = "order_purchase_timestamp")[
    "order_id"].count()

# Visualization
sns.relplot(data = orders_date,
            kind = "line")
plt.xticks(rotation = 90)
plt.annotate("Black Friday", xy = (
    pd.Timestamp("2017-12-10"), 1100))
plt.xlabel("Date")
plt.ylabel("Nº of Orders")
plt.title("Daily Orders", pad = pad)
plt.show()

# Identify max
orders_date[orders_date == orders_date.max()]



# ----------------------------------------
## Notes ##

# The Delivered Status represent around 97% of all orders.
# - Missing dates in the orders dataset can be explained by the order status.



# ----------------------------------------
## Pickle ##

file_path = "orders.pkl"
df = orders

# Open the file in binary write mode ('wb')
with open(file_path, 'wb') as file:
    # Use pickle.dump() to serialize the DataFrame and write it to the file
    pickle.dump(df, file)



# --------------------------------------------------------------------
# ### Payments Dataset ###

payments = pd.read_csv("olist_order_payments_dataset.csv")



# ----------------------------------------
## Dataset Inspection ##

payments.info()
payments.iloc[:,1:].head()    # the first column is ID



# ----------------------------------------
## Data Range Constraints ##

payments.describe()

# payment_value
# Visualization 
sns.displot(
    data = payments,
    x = "payment_value",
    kind = "hist")
plt.show()


# payment_sequential
# Understand the maximum of 29
payments[payments["payment_sequential"] == 29]    # to get the order_id
payments[payments["order_id"] == "fa65dad1b0e818e3ccc5cb0e39231352"]["payment_type"]



# ----------------------------------------
## Uniqueness Constraints ##

# Checking for duplicates in the dataset
payments.duplicated().any()


# Checking for duplicates in id
payments["order_id"].duplicated().any()

# Checking for an example
payments["order_id"].value_counts().head()

payments[
    payments["order_id"] == "fa65dad1b0e818e3ccc5cb0e39231352"].iloc[:,1:]



# ----------------------------------------
## Categorical Variables ##

# Identify possible category features
payments.nunique()

# Nº of Payments per Type
n_pay_type = payments["payment_type"].value_counts(normalize = True)

# Visualization
sns.catplot(
    data = n_pay_type,
    kind = "bar")
plt.title("Proportion of Payments per Type", pad = pad)
plt.xticks(rotation = 45)
plt.show()


# Checking "not_defined" type
payments[payments["payment_type"] == "not_defined"]

# As they are only 3 orders and have no payment value, they will be dropped
ind_drop = payments[payments["payment_type"] == "not_defined"].index
payments.drop(index = ind_drop, inplace = True)



# ----------------------------------------
## Completeness ##

payments.isna().sum()



# ----------------------------------------
## Analysis ##

sns.catplot(data = payments,
            x = "payment_installments",
            y = "payment_value",
            kind = "bar",
            ci = False)
plt.xlabel("Nº of Installments")
plt.ylabel("Payment Value")
plt.title("Payment Value per Nº of Installments", pad = pad)
plt.xticks(rotation = 90)
plt.show()


# ----------------------------------------
## Notes ##

# - There are duplicated order_id due to different payment types and 
# installments.
# - payment_sequential refers to different payment_types for the same order. 
# For example, a payment with 2 payments types like "voucher" and 
# "credit card" will have 2 installments. However, a payment with 2 uses of
# "voucher" will also have 2 installments.



# ----------------------------------------
## Pickle ##

file_path = "payments.pkl"
df = payments

# Open the file in binary write mode ('wb')
with open(file_path, 'wb') as file:
    # Use pickle.dump() to serialize the DataFrame and write it to the file
    pickle.dump(df, file)



# --------------------------------------------------------------------
# ### Products Dataset ###

products = pd.read_csv("olist_products_dataset.csv")



# ----------------------------------------
## Dataset Inspection ##

products.info()
products.iloc[:,1:].head()    # the first two column is ID



# ----------------------------------------
## Data Range Constraints ##

products.describe()



# ----------------------------------------
## Uniqueness Constraints ##

# Checking for duplicates in the dataset
products.duplicated().any()


# Checking for duplicates in id
products["product_id"].duplicated().any()



# ----------------------------------------
## Categorical Variables ##

# Identify possible category features
products.nunique()

# Categories
products["product_category_name"].unique()

# Visualization
top10_prod_cat = products["product_category_name"].value_counts(
    normalize = True).head(10)

# Visualization
sns.catplot(
    data = top10_prod_cat ,
    kind = "bar")
plt.xticks(rotation = 90)
plt.title("Proportion of  Top 10 Categories by Products", pad = pad)
plt.show()



# ----------------------------------------
## Completeness ##

products.isna().sum()



# ----------------------------------------
## Notes ##
# - The only relevant feature in this dataset for our churn case, would be the
# categories. However, there is only 1.20 products on average per order, which
# means, there is no diversity of categories on each order.



# ----------------------------------------
## Pickle ##

file_path = "products.pkl"
df = products

# Open the file in binary write mode ('wb')
with open(file_path, 'wb') as file:
    # Use pickle.dump() to serialize the DataFrame and write it to the file
    pickle.dump(df, file)



# --------------------------------------------------------------------
# ### Order Items Dataset ###

order_items = pd.read_csv("olist_order_items_dataset.csv")



# ----------------------------------------
## Dataset Inspection ##

order_items.info()
order_items.iloc[:,1:].head()    # the first two column is ID



# ----------------------------------------
## Data Range Constraints ##

order_items.describe()

# price and freight_value
# Visualization 
sns.displot(
    data = order_items,
    x = "price",
    kind = "hist")
plt.show()

sns.displot(
    data = order_items,
    x = "freight_value",
    kind = "hist")
plt.show()


# Freight Value
# Upper Outliers

# Quantiles
freight_q75 = order_items["freight_value"].quantile(0.75)
freight_q25 = order_items["freight_value"].quantile(0.25)
freight_iqr = freight_q75 - freight_q25
freight_upper_out = freight_q75 + (1.5 * freight_iqr)

# Proportion of outliers
# There are repeated products
prop_freight_valu_outliers = len(
    order_items[order_items["freight_value"] > freight_upper_out]) / len(
        order_items)

        
# Checking dimensions of outliers
# Identify product_id
freight_outliers_df = order_items[order_items["freight_value"] > freight_upper_out]

freight_outliers_prod_productid = freight_outliers_df[
    "product_id"].to_list()

# New DF
prod_freight_val_analysis_df = products.assign(
    freight_outlier = np.where(products["product_id"].isin(
        freight_outliers_prod_productid), 1, 0))

prod_freight_val_analysis_df["product_vol"] = prod_freight_val_analysis_df[
    "product_length_cm"] * prod_freight_val_analysis_df[
        "product_height_cm"] * prod_freight_val_analysis_df[
            "product_width_cm"]

# Visualization
sns.relplot(data = prod_freight_val_analysis_df,
            x = "product_weight_g",
            y = "product_vol",
            hue = "freight_outlier",
            palette = [blue_color] + [insight_color],
            kind = "scatter")
plt.show()

sns.catplot(data = prod_freight_val_analysis_df,
            x = "freight_outlier",
            y = "product_weight_g",
            kind = "box")
plt.title("Product Volume by Freight Value Outlier Groups")
plt.ylabel("Product Volume")
plt.xlabel("")
plt.xticks([0,1], labels = ["No Outlier", "Outlier"])
plt.show()

# Pivot Table
freight_pivot = pd.pivot_table(data = prod_freight_val_analysis_df,
                               index = "freight_outlier",
                               values = ["product_weight_g", "product_vol"],
                               aggfunc = "mean")

# Heatmap
sns.heatmap(data = freight_pivot,
            annot = True,
            fmt = ".2f",
            xticklabels = ["Product Volume", "Product Weight"],
            yticklabels = ["Not Outlier", "Outlier"],
            cmap = "Blues")
plt.ylabel("")
plt.title("Average Values per Freight Value Outlier Groups", pad = pad)
plt.show()

# Hypothesis Test
# H0: The distribution of product volume is the same for the Freight 
# Value Outliers group and the Not Outliers group.
# H1: The distribution of product volume for the Freight Value Outliers 
# group is stochastically greater than the distribution of product volume for 
# the Not Outliers group.

prod_freight_val_analysis_df_na = prod_freight_val_analysis_df.dropna()

u_stat, u_pval = mannwhitneyu(
    prod_freight_val_analysis_df_na[prod_freight_val_analysis_df_na[
        "freight_outlier"] == 1]["product_vol"],
    prod_freight_val_analysis_df_na[prod_freight_val_analysis_df_na[
        "freight_outlier"] == 0]["product_vol"],
    alternative = "greater")

print(f"Mann-Whitney U test p-value: {u_pval:.4f}")

# Hypothesis Test
# H0: The distribution of product weight is the same for the Freight Value 
# Outliers group and the Not Outliers group.
# H1: The distribution of product weight for the Freight Value Outliers 
# group is stochastically greater than the distribution of product volume for the Not Outliers group.

u_stat, u_pval = mannwhitneyu(
    prod_freight_val_analysis_df_na[prod_freight_val_analysis_df_na[
        "freight_outlier"] == 1]["product_weight_g"],
    prod_freight_val_analysis_df_na[prod_freight_val_analysis_df_na[
        "freight_outlier"] == 0]["product_weight_g"],
    alternative = "greater")

print(f"Mann-Whitney U test p-value: {u_pval:.4f}")



# Monetary Value
# Upper Outliers

# Quantiles
price_q75 = order_items["price"].quantile(0.75)
price_q25 = order_items["price"].quantile(0.25)
price_iqr = price_q75 - price_q25
price_upper_out = price_q75 + (1.5 * price_iqr)

# Proportion of outliers
# There are repeated products
prop_price_outliers = len(
    order_items[order_items[
        "price"] > price_upper_out]) / len(order_items)

        
# Checking categories of outliers
# Identify product_id
price_outliers_df = order_items[order_items["price"] > price_upper_out]
price_outliers_df.sort_values(by = "price", ascending = False).iloc[:, 2:].head()


# Checking Categories
price_outl_productid = price_outliers_df["product_id"].to_list()

price_out_products_df = products[products["product_id"].isin(
    price_outl_productid )]

price_out_categories = (price_out_products_df["product_category_name"].value_counts(
    normalize = True).head(10) * 100).reset_index(name = "prop")

# Visualization
sns.catplot(data = price_out_categories,
            x = "prop",
            y = "product_category_name",
            kind = "bar")
plt.title("Price Outliers per Category", pad = pad)
plt.xlabel("Proportion (%)")
plt.ylabel("Top 10 Categories")
plt.show()


# Check notes section for outliers analysis


# ----------------------------------------
## Uniqueness Constraints ##

# Checking for duplicates in the dataset
order_items.duplicated().any()


# Checking for duplicates in id
order_items["order_id"].duplicated().any()

# Checking for an example
order_items[order_items["order_id"].duplicated(keep = False)].head()



# ----------------------------------------
## Categorical Variables ##

# Identify possible category features
order_items.nunique()



# ----------------------------------------
## Completeness ##

order_items.isna().sum()



# ----------------------------------------
## Analyse ##

quant_products = order_items.groupby("order_id")["order_item_id"].max()
quant_average = quant_products.mean()


# ----------------------------------------
## Notes ##

# - price and freight_value features are skewed.
# - After checking boxplot and scatterplot for freight value outliers, it was 
# possible to see they are mainly related to heavy and big products. 
# With the current info, they do not look errors and will be kept. We would 
# need to check further with the company, to be sure about errors.
# - Price outliers are dispersed through several categories. We would need
# to check further with the company if there are errors. With the present
# information, we have no evidence of errors, so we will keep them. 
# - order_id may be duplicated due to different quantities of the same product
# showed as "order_item_id"
# - The average number of products per order is 1.20.



# ----------------------------------------
## Pickle ##

file_path = "order_items.pkl"
df = order_items

# Open the file in binary write mode ('wb')
with open(file_path, 'wb') as file:
    # Use pickle.dump() to serialize the DataFrame and write it to the file
    pickle.dump(df, file)



# --------------------------------------------------------------------
# ### Sellers Dataset ###

sellers = pd.read_csv("olist_sellers_dataset.csv")



# ----------------------------------------
## Dataset Inspection ##

sellers.info()
products.iloc[:,1:].head()    # the first two column is ID



# ----------------------------------------
## Data Range Constraints ##

sellers.describe()



# ----------------------------------------
## Uniqueness Constraints ##

# Checking for duplicates in the dataset
sellers.duplicated().any()


# Checking for duplicates in id
sellers["seller_id"].duplicated().any()



# ----------------------------------------
## Categorical Variables ##

# Identify possible category features
sellers.nunique()


# Nº of clients per State
n_sel_state = sellers["seller_state"].value_counts(normalize = True).head(10)

# Visualization
sns.catplot(
    data = n_sel_state,
    kind = "bar")
plt.title("Proportion of Sellers per State", pad = pad)
plt.xlabel("Top 10 States")
plt.show()



# ----------------------------------------
## Completeness ##

sellers.isna().sum()



# ----------------------------------------
## Notes ##
# - SP represents almost 60% of total sellers (aligned with proportion of
# clients).



# ----------------------------------------
## Pickle ##

file_path = "sellers.pkl"
df = sellers

# Open the file in binary write mode ('wb')
with open(file_path, 'wb') as file:
    # Use pickle.dump() to serialize the DataFrame and write it to the file
    pickle.dump(df, file)



# --------------------------------------------------------------------
# ### Reviews Dataset ###

reviews = pd.read_csv("olist_order_reviews_dataset.csv")



# ----------------------------------------
## Dataset Inspection ##

reviews.info()
reviews.iloc[:,2:].head()    # the first two column is ID



# ----------------------------------------
## Data Range Constraints ##

reviews.describe()



# ----------------------------------------
## Uniqueness Constraints ##

# Checking for duplicates in the dataset
reviews.duplicated().any()


# Checking for duplicates in payment_id
reviews["review_id"].duplicated().any()

# Checking for an example
reviews["review_id"].value_counts().head()

reviews[
    reviews["review_id"] == "dbdf1ea31790c8ecfcc6750525661a9b"]



# ----------------------------------------
## Categorical Variables ##

# Identify possible category features
reviews.nunique()

# review_score type is int, but we will analyze it here
# Nº of Reviews per Score
n_rev_score = reviews["review_score"].value_counts(normalize = True)

# Visualization
sns.catplot(
    data = n_rev_score ,
    kind = "bar")
plt.title("Proportion of Reviews per Score")
plt.show()



# ----------------------------------------
## Completeness ##

reviews.isna().sum()



# ----------------------------------------
## Analyze ##

# Duplicated review_id for different order_id
# Getting order_id that have different 
dupl_reviews = reviews.loc[
    reviews["review_id"].duplicated(keep = False),
    ["review_id", "order_id"]].sort_values(by = "review_id")

# Inspect
dupl_reviews.head()


# Check one example of two order_id with same review_id
review_examples = ["dfcdfc43867d1c1381bfaf62d6b9c195", "04a28263e085d399c97ae49e0b477efa"]


# On orders dataset
orders[orders["order_id"].isin(review_examples)].iloc[:, 3:]
# order_purchase_timestamp is only 2 seconds different


# On order_items dataset
order_items[order_items["order_id"].isin(review_examples)].iloc[:, 3:]
# Different product_id and seller_id



# ----------------------------------------
## Notes ##

# - There are duplicated review_id for different order_id dut to different
# products and sellers.
# - There are missing data for comments, but they are not relevant for this
# project.



# ----------------------------------------
## Pickle ##

file_path = "reviews.pkl"
df = reviews

# Open the file in binary write mode ('wb')
with open(file_path, 'wb') as file:
    # Use pickle.dump() to serialize the DataFrame and write it to the file
    pickle.dump(df, file)
