# -*- coding: utf-8 -*-
"""
Project - Brazilian E-Commerce Public Dataset by Olist (Retention Prediction)

EDA and Data Preparation

@author: Patrícia Pereira

"""


# --------------------------------------------------------------------
# ### Import Packages ###

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
from scipy.stats import pearsonr, spearmanr



# --------------------------------------------------------------------
# ### Define visualization style ###

sns.set(font_scale = 1.4)
sns.set_style("ticks")
sns.set_palette("winter")

blue_color = "#0000FF"
insight_color = "#00FFFF"



# --------------------------------------------------------------------
# ### Load features_analysis_df from Initial Data Prepation py file ###


with open('features_analysis_df.pkl', 'rb') as file:
    features = pickle.load(file)



# --------------------------------------------------------------------
# ### EDA for Retention Prediction ###


# ----------------------------------------
## Inspect ##

features.info()



# ----------------------------------------
## customer_state ##

# Aggregation
top_10_state = (features[
    "customer_state"].value_counts(normalize = True) * 100).head(10)

top_10_state_list = top_10_state.index

top_10_state_df = features[features["customer_state"].isin(top_10_state_list)]

top_10_state_target = (top_10_state_df.groupby("customer_state")[
    "target"].value_counts(
        normalize = True) * 100).reset_index(
            name = "prop_%").rename(
                columns = {"target": "Retained"})


# Visualization
custom_palette_state = [insight_color] + [blue_color] * (len(top_10_state) - 1)

sns.catplot(
    data = top_10_state,
    kind = "bar",
    palette = custom_palette_state)
plt.title("Proportion of Clients per State", pad = 15, )
plt.xlabel("Top 10 States")
plt.ylabel("Proportion (%)")
plt.show()                
                
                
# Visualization - FacetGrid
sns.catplot(data = top_10_state_target,
            x = "customer_state",
            y = "prop_%",
            hue = "Retained",
            palette = [blue_color] + [insight_color],
            kind = "bar",
            dodge = False)
plt.title("Proportion of Retained Customers by State", pad = 15)
plt.ylabel("Proportion (%)")
plt.xlabel("")
plt.show()


# Hypothesis Test
# Chi-square Test for independency between these two categorical variables.
# H0: There is independence between the two categorical variables
contingency_table = pd.crosstab(features["customer_state"],
                                features["target"])

chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

print(f"Chi2-statistic: {chi2_stat}, P-value: {p_val}")
# The high p-value suggests that we do not have sufficient statistical 
# evidence to conclude that there is a significant association between 
# customer state and retention rate.



# ----------------------------------------
## avg_diff_days_del_purch and avg_diff_days_estdel_del ##
diff_days_df = features[["avg_diff_days_estdel_del",
                         "avg_diff_days_del_purch",
                         "avg_review_score",
                         "target"]].rename(
                             columns = {"avg_review_score": "Review Score",
                             "target": "Retained"})

# Visualization
g = sns.relplot(data = diff_days_df,
            y = "avg_diff_days_estdel_del",
            x = "avg_diff_days_del_purch",
            hue = "Review Score",
            palette = "Blues",
            size = "Review Score",
            col = "Retained")
g.set_xlabels("Average Delivery Waiting Time (Days)")
g.set_ylabels("Diff. Estimated and Actual Delivery (Days)")
plt.show()


# Both features look correlated
# Correlation
features[["avg_diff_days_estdel_del", "avg_diff_days_del_purch"]].corr()

# Hypothesis Test
# H0: The correlation coefficient is equal to zero.
# H1:  The correlation coefficient is not equal to zero. This means there 
# is a linear relationship (either positive or negative) between the two 
# variables in the population.
p_stat, p_pval = pearsonr(features["avg_diff_days_estdel_del"],
         features["avg_diff_days_estdel_del"])

print(f"Pearson correlation coefficient test p-value: {p_pval.round(4): }")
# As the p-value is 0, we reject the null hypothesis that the correlation 
# coefficient is equal to zero.



# ----------------------------------------
## avg_review_score ##


# Aggregation
features.groupby("target")["avg_review_score"].mean()


# Visualization
sns.displot(data = features,
            x = "avg_review_score",
            kind = "kde")
plt.show()


# Hypothesis test
# No normal distribution
# Assuming indepency between retained and no retained customers

# Dropping missing values first
feat_rev_cleaned = features.dropna(subset = "avg_review_score")

# H0: The mean review score is the same
# H1: The mean review is higher for retained customers
u_stat, u_pval = mannwhitneyu(
    feat_rev_cleaned[feat_rev_cleaned["target"] == 1]["avg_review_score"],
    feat_rev_cleaned[feat_rev_cleaned["target"] == 0]["avg_review_score"],
    alternative = "greater")

print(f"Mann-Whitney U test p-value: {u_pval.round(4): }")
# As the p-value < 5%, we reject the null hypothesis that the means are the 
# same and have statistical evidence that the mean for retained customers is
# higher.



# ----------------------------------------
## Freight Value and Freight Rate ##


# Describe
features["avg_freight_value"].describe()
features["avg_freight_rate"].describe()

# Visualization
sns.displot(data = features,
            x = "avg_freight_value",
            kind = "hist")
plt.show()

sns.displot(data = features,
            x = "avg_freight_rate",
            kind = "hist")
plt.show()


# As mentioned on Initial EDA, it looks like the freight value outliers 
# are related to heavy and big products. Therefore, they will be kept in the
# dataset.


# Aggregation
target_freight = features.groupby("target")["avg_freight_value"].mean()
features.groupby("target")["avg_freight_rate"].mean()

# Visualization
sns.catplot(data = features,
            y = "avg_freight_value",
            x = "target",
            kind = "point")
plt.title("Average Freight Value per Customer", pad = 15)
plt.xlabel("Retained Customer")
plt.ylabel("Average Freight Value")
plt.show()

sns.displot(data = features,
            x = "avg_freight_value",
            hue = "target",
            kind = "kde")
plt.show()

# Hypothesis test
# No normal distribution
# Assuming indepency between retained and no retained customers

# H0: The mean freight value is the same
# H1: The mean freight value is higher for non-retained customers
u_stat, u_pval = mannwhitneyu(
    features[features["target"] == 0]["avg_freight_value"],
    features[features["target"] == 1]["avg_freight_value"],
    alternative = "greater")

print(f"Mann-Whitney U test p-value: {u_pval.round(4): }")
# As the p-value is high, we cannot reject the null hypothesis that the 
# means are the same and have no statistical evidence that the mean for 
# non-retained customers is higher.


# H0: The mean freight rate is the same for retained and non-retained cust.
# H1: The mean freight rate is different
u_stat, u_pval = mannwhitneyu(
    features[features["target"] == 0]["avg_freight_rate"],
    features[features["target"] == 1]["avg_freight_rate"])

print(f"Mann-Whitney U test p-value: {u_pval.round(4): }")
# As the p-value is high, we cannot reject the null hypothesis that the 
# means are the same and have no statistical evidence that the mean for 
# non-retained customers is different.



# --------------------------------------------------------------------
# ### EDA for Customer Segmentation ###


# ----------------------------------------
## RFM and Tenure ##


# Visualization
fig, ax = plt.subplots(2,2, figsize = (11, 8))

sns.histplot(data = features,
            x = "recency",
            ax = ax[0,0])
sns.histplot(data = features,
            x = "frequency",
            ax = ax[0,1])
sns.histplot(data = features,
            x = "monetary_value",
            ax = ax[1,0])
sns.histplot(data = features,
            x = "tenure",
            ax = ax[1,1])

plt.show()


# Frequency
features["frequency"].value_counts(normalize = True)
features["frequency"].mean()
freq_str = features["frequency"].astype("str")
freq_str_cat = freq_str.map({"1": "1"}).fillna("> 1")

sns.catplot(data = freq_str_cat,
            kind = "count",
            stat = "percent")
plt.show()


# Correlation
features[["recency", 
          "tenure"]].corr()
# There is correlation between these two variables. 
# This is due to the fact that frequency is only 1.02.



# ----------------------------------------
## Sellers and Products ##

features["tt_unique_sellers"].value_counts(normalize = True)
features["tt_unique_products"].value_counts(normalize = True)
features["avg_quant_prod"].value_counts(normalize = True)

# As frequency shows, around 98% of customers only bought once. Therefore,
# these features are not relevant for further analysis.


# Correlation
feat_corr = features[["frequency", 
          "tt_unique_sellers",
          "tt_unique_products",
          "avg_quant_prod",
          "tenure",
          "recency"]]

sns.pairplot(feat_corr)
plt.show()

# Considering Spearman instead of Pearson Correlation due to non-linearity
# and suitable for discrete data

sns.heatmap(feat_corr.corr(method = "spearman"),
            annot = True,
            fmt = ".2f",
            cmap='Blues')
plt.title("Spearman Correlation", pad = 15)
plt.show()


# Hypothesis Test
# H0: The correlation coefficient is equal to zero.
# H1:  The correlation coefficient is not equal to zero. This means there 
# is a linear relationship (either positive or negative) between the two 
# variables in the population.
s_stat, s_pval = spearmanr(features["frequency"],
         features["tt_unique_sellers"])

print(f"Spearman correlation coefficient test p-value: {s_pval.round(4): }")
# As the p-value is 0, we reject the null hypothesis that the correlation 
# coefficient is equal to zero.

s_stat, s_pval = spearmanr(features["frequency"],
         features["tt_unique_products"])

print(f"Spearman correlation coefficient test p-value: {s_pval.round(4): }")
# As the p-value is 0, we reject the null hypothesis that the correlation 
# coefficient is equal to zero.



# ----------------------------------------
## Payment type and Nº of Installments ##


# Visualization
sns.displot(data = features,
            x = "avg_n_installments",
            kind = "kde")
plt.show()


# Visualization
sns.catplot(data = features,
            x = "pref_pay_type",
            kind = "count",
            stat = "percent")
plt.show()

sns.catplot(data = features,
            x = "pref_pay_type",
            y = "avg_n_installments",
            kind = "box")
plt.title("Nº of Installments per Type of Payment", pad = 15)
plt.ylabel("Average Nº of Installments")
plt.xlabel("Preferred Type of Payment")
plt.xticks(ticks = ["credit_card", "boleto", "voucher", "debit_card"],
           labels = ["Credit Card", "Boleto", "Voucher", "Debit Card"])
plt.show()

sns.catplot(data = features,
            x = "pref_pay_type",
            y = "monetary_value",
            kind = "box")
plt.show()


# Aggregation
features.groupby("pref_pay_type")["monetary_value"].describe().T
features.groupby("pref_pay_type")["avg_n_installments"].describe().T


# Hypothesis test
# No normal distribution

# H0: The mean monetary value is the same for different payment types
# H1: At least two payment types have significantly different monetary value
k_stat, k_pval = kruskal(
    features[features["pref_pay_type"] == "credit_card"]["monetary_value"],
    features[features["pref_pay_type"] == "boleto"]["monetary_value"],
    features[features["pref_pay_type"] == "voucher"]["monetary_value"],
    features[features["pref_pay_type"] == "debit_card"]["monetary_value"])
    
print(f"Kruskal-Wallis test p-value: {k_pval.round(4): }")
# As the p-value is 0, we can reject the null hypothesis that the 
# mean monetary value is the same for different payment types and have 
# statistical evidence that at least two payment types have significantly 
# different monetary value.


# H0: The mean nº of installments is the same for different payment types
# H1: At least two payment types have significantly different nº of 
# installments
k_stat, k_pval = kruskal(
    features[features["pref_pay_type"] == "credit_card"]["avg_n_installments"],
    features[features["pref_pay_type"] == "boleto"]["avg_n_installments"],
    features[features["pref_pay_type"] == "voucher"]["avg_n_installments"],
    features[features["pref_pay_type"] == "debit_card"]["avg_n_installments"])
    
print(f"Kruskal-Wallis test p-value: {k_pval.round(4): }")
# As the p-value is 0, we can reject the null hypothesis that the 
# mean nº of installments is the same for different payment types and have 
# statistical evidence that at least two payment types have significantly 
# different mean nº of installments.



# ----------------------------------------
## Customer State ##


# Customer State per Monetary Value
features.groupby("customer_state")[
    "monetary_value"].mean().sort_values(ascending = False).head(10)

# Visualization 
custom_palette_state_mon_value = [blue_color] * (len(top_10_state))
custom_palette_state_mon_value[9] = insight_color

sns.catplot(data = top_10_state_df,
            x = "customer_state",
            y = "monetary_value",
            palette = custom_palette_state_mon_value,
            kind = "bar",
            order = top_10_state.index)
plt.title("Average Monetary Value per State", pad = 15)
plt.ylabel("Monetary Value")
plt.xlabel("Top 10 States")
plt.show()


# Aggregation
top_10_state_df.groupby("customer_state")["monetary_value"].mean()
features["customer_state"].value_counts(normalize = True)

# SP has the most proportion and DF and GO has the higher average monetary 
# value. Therefore we will consider both for a categorical variable.



# --------------------------------------------------------------------
# ### Correlation ###


features_num = features.select_dtypes("number")

sns.heatmap(features_num.corr(method = "spearman"),
            annot = True,
            fmt = ".2f",
            cmap='Blues')
plt.title("Spearman Correlation", pad = 15)
plt.show()



# --------------------------------------------------------------------
# ### Final Features Table ###


# ----------------------------------------
## Create New Features ##

features["SP_DF"] = features["customer_state"].map(
    {"SP": "SP",
     "DF": "DF"}).fillna("other_state")


# ----------------------------------------
## Drop irrelevant Features ##

# Features to drop due to previous analysis
features_to_drop = ["customer_state",    # new categorical variable
                    "avg_diff_days_estdel_del",    # high correlation
                    "avg_freight_value",    # hypothesis test
                    "avg_freight_rate",    # hypothesis test
                    "tenure",    # high correlation
                    "tt_unique_sellers",    # high correlation
                    "tt_unique_products"]    # high correlation    

# Final Table
feature_table = features.drop(features_to_drop, axis = 1)

# Inspect
feature_table.info()


# ----------------------------------------
## Correlation ##

feature_table_num = feature_table.select_dtypes("number")

sns.heatmap(feature_table_num.corr(method = "spearman"),
            annot = True,
            fmt = ".2f",
            cmap='Blues')
plt.title("Spearman Correlation", pad = 15)
plt.show()



# --------------------------------------------------------------------
# ### Pickle: feature_table ###


# Transform feature_table to pickle file for modeling

file_path = "feature_table_kmeans.pkl"
df = feature_table

# Open the file in binary write mode ('wb')
with open(file_path, 'wb') as file:
    # Use pickle.dump() to serialize the DataFrame and write it to the file
    pickle.dump(df, file)
