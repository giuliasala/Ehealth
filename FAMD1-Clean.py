#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install prince


# In[2]:


pip install scikit-learn-extra


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import prince
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import davies_bouldin_score
from scipy import stats
from prince import MCA
from statsmodels.stats.multitest import multipletests



# In[4]:


df=pd.read_csv("/Users/mohamedshoala/Documents/third semester/e health methods and applications/project/project data/dataset_project_eHealth20232024.csv")
print(df.info)
print(f'nan in the df: {df.isnull().sum().sum()}')
print(f'rows with at least 1 nan: {df.isnull().T.any().T.sum()}')
print(f'columns with at least 1 nan: \n{df.isnull().any()}')



# In[5]:


# For other columns, replace missing values with the respective column's median
for column in df.columns:
    median_value = df[column].median()
    df[column].fillna(value=median_value, inplace=True)



# In[6]:


print(f'nan in the df: {df.isnull().sum().sum()}')


# In[7]:


df.drop_duplicates(keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.info)



# In[8]:


col_phq = ['phq_1', 'phq_2', 'phq_3', 'phq_4', 'phq_5', 'phq_6', 'phq_7', 'phq_8', 'phq_9']
sum_phq = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_phq].sum()
    sum_phq.append(row_sum)
print(f'phq sum: {sum_phq}')
# threshold is 10 (over=depression)



# In[9]:


col_gad = ['gad_1', 'gad_2', 'gad_3', 'gad_4', 'gad_5', 'gad_6', 'gad_7']
sum_gad = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_gad].sum()
    sum_gad.append(row_sum)
print(f'gad sum: {sum_gad}')
# threshold is 10 (over=anxious)



# In[10]:


col_eheals = ['eheals_1', 'eheals_2', 'eheals_3', 'eheals_4', 'eheals_5', 'eheals_6', 'eheals_7', 'eheals_8']
sum_eheals = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_eheals].sum()
    sum_eheals.append(row_sum)
print(f'eheals sum: {sum_eheals}')
# no threshold, its subjective



# In[11]:


col_heas = ['heas_1', 'heas_2', 'heas_3', 'heas_4', 'heas_5', 'heas_6', 'heas_7', 'heas_8', 'heas_9', 'heas_10', 'heas_11', 'heas_12', 'heas_13']
sum_heas = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_heas].sum()
    sum_heas.append(row_sum)
print(f'heas sum: {sum_heas}')
# we have to find the threshold!



# In[12]:


# reverse score of ccs 3, 6, 7, 12
df['ccs_3'] = df['ccs_3'].max() - df['ccs_3']
df['ccs_6'] = df['ccs_6'].max() - df['ccs_6']
df['ccs_7'] = df['ccs_7'].max() - df['ccs_7']
df['ccs_12'] = df['ccs_12'].max() - df['ccs_12']

col_ccs = ['ccs_1', 'ccs_2', 'ccs_3', 'ccs_4', 'ccs_5', 'ccs_6', 'ccs_7', 'ccs_8', 'ccs_9', 'ccs_10', 'ccs_11', 'ccs_12']
sum_ccs = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_ccs].sum()
    sum_ccs.append(row_sum)
    # high score=climate change skeptic
print(f'ccs sum: {sum_ccs}')
# we have to find the threshold!



# In[13]:


# creating the new dataframe
df1 = df[[ 'gender', 'education', 'marital']]
data2 = {
    'phq': sum_phq,
    'gad': sum_gad,
    'eheals': sum_eheals,
    'heas': sum_heas,
    'ccs': sum_ccs
}
df2 = pd.DataFrame(data2)
df_sum = pd.concat([df1, df2], axis=1)
df_sum



# In[14]:


# Replace 'column1' and 'column2' with the actual names of your numerical columns.
df_sum['gad'] = df_sum['gad'].apply(lambda x: 1 if x >= 10 else 0)
df_sum['phq'] = df_sum['phq'].apply(lambda x: 1 if x >= 5 else 0)


# In[15]:


# Define the bins and labels for the 'education' column
bins = [4, 7, 12, 17, 21, 24, 28]  # One more than the labels to define the edges of the bins
labels = ['Elementary', 'Middle', 'High', 'Bachelors', 'Masters', 'Doctoral']

# Use pandas.cut to bin the data into categories
df_sum['education_level'] = pd.cut(df_sum['education'], bins=bins, labels=labels, include_lowest=True, right=False)

# Check the binning result
print(df_sum['education_level'].value_counts())


# In[16]:


df_numerical1=df[["age","income"]]
df_numerical2=df_sum[["eheals","heas","ccs"]]
df_numerical=pd.concat([df_numerical1, df_numerical2], axis=1)
df_numerical


# In[17]:


columns_to_drop = ['eheals', 'heas',"ccs"]
df_sum = df_sum.drop(columns=columns_to_drop)
df_sum1=df_sum
df_sum1


# In[18]:


columns_to_drop = ['education']
df_sum = df_sum.drop(columns=columns_to_drop)
df_sum


# In[19]:


# Apply one-hot encoding to the 'education_level' column
education_dummies = pd.get_dummies(df_sum['education_level'], prefix='edu')

# Concatenate the one-hot encoded columns back to the original dataframe
df_sum = pd.concat([df_sum, education_dummies], axis=1)


# In[20]:


df_sum


# In[21]:


columns_to_drop = ['education_level']
df_sum = df_sum.drop(columns=columns_to_drop)
df_sum


# In[22]:


from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

encoder = OneHotEncoder(sparse_output=False)  # Updated parameter name
categorical_columns = ['gender', 'marital']  # replace with your actual categorical columns
categorical_encoded = encoder.fit_transform(df_sum[categorical_columns])
categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_columns))


# In[23]:


df_sum = pd.concat([df_sum, categorical_encoded_df], axis=1)
df_sum.drop('marital', axis=1, inplace=True)
df_sum.drop('gender', axis=1, inplace=True)

df_sum


# In[24]:


import pandas as pd

# Assuming you have your original DataFrame as df_sum
min_values = df_sum.min()
max_values = df_sum.max()

# Combine min and max into a single DataFrame for a clearer overview
min_max_values = pd.DataFrame({'Min': min_values, 'Max': max_values})

print(min_max_values)


# In[25]:


import pandas as pd

# Assuming you have your original DataFrame as df_sum
min_values = df_numerical.min()
max_values = df_numerical.max()

# Combine min and max into a single DataFrame for a clearer overview
min_max_values = pd.DataFrame({'Min': min_values, 'Max': max_values})

print(min_max_values)


# In[26]:


# Create the scaler objects
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

# Fit and transform the columns with the StandardScaler
df_numerical[['age', 'income']] = scaler_standard.fit_transform(df_numerical[['age', 'income']])

# Fit and transform the columns with the MinMaxScaler
df_numerical[['eheals', 'heas', 'ccs']] = scaler_minmax.fit_transform(df_numerical[['eheals', 'heas', 'ccs']])





# Combine the scaled continuous and one-hot encoded categorical data
#df_scaled = pd.concat([categorical_encoded_df, continuous_scaled_df], axis=1)




# In[27]:


df_numerical


# In[28]:


#BOX_PLOT
get_ipython().run_line_magic('matplotlib', 'inline')
df_numerical.boxplot(rot=90)


# In[29]:


sns.pairplot(df_numerical)
plt.show()


# In[30]:


#so after scaling with min_max scaler we still have the same representation of all the categories 


# In[31]:


# Calculate the medians for each column in the original DataFrame
medians = df_numerical.median()

# List of columns you want to process
columns_to_process = ['income']  # Add the names of the columns you want to process
threshold = 1.5

# Create a copy of the DataFrame without outliers
df_no_outliers = df_numerical.copy()

for column in columns_to_process:
    z_scores = np.abs(stats.zscore(df_no_outliers[column]))
    outliers = (z_scores > threshold)

    # Replace outliers with the median value of the column
    df_no_outliers[column][outliers] = medians[column]


# In[32]:


df_no_outliers


# In[33]:


#BOX_PLOT
get_ipython().run_line_magic('matplotlib', 'inline')
df_no_outliers.boxplot(rot=90)


# In[34]:


if df_no_outliers.isna().values.any():
    print("There are NaN values in the DataFrame.")
else:
    print("No NaN values in the DataFrame.")


# In[35]:


df_all = pd.concat([df_sum, df_no_outliers], axis=1)
df_all


# In[36]:


#BOX_PLOT
get_ipython().run_line_magic('matplotlib', 'inline')
df_all.boxplot(rot=90)


# In[37]:


# elbow method
inertia = []  # empty list to store the sum of squared distances (inertia) for different K values
k_values = range(1, 11)
for k in k_values:
     kmedoids = KMedoids(n_clusters=k, random_state=0)
     kmedoids.fit(df_all)
     inertia.append(kmedoids.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid()
plt.show()


# In[38]:


from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

# Specify the number of clusters (K) and the maximum number of iterations (max_iter)
k = 3
max_iter = 100

# Create a KMedoids instance with a fixed random state
kmedoids = KMedoids(n_clusters=k, max_iter=max_iter, random_state=0)

# Fit the data, ensure df_all is not changing between runs
kmedoids.fit(df_all)

# Get the cluster assignments
cluster_labels = kmedoids.labels_

# Add the cluster labels to the DataFrame
df_all['Cluster'] = cluster_labels

# Make sure to exclude the 'Cluster' column when calculating the silhouette score
features_for_scoring = df_all.drop('Cluster', axis=1)
silhouette_avg = silhouette_score(features_for_scoring, cluster_labels)

print(f"Silhouette Score: {silhouette_avg}")


# In[39]:


df_all


# In[40]:


df_famd=pd.concat([df_sum1, df_no_outliers], axis=1)
df_famd = df_famd.drop('education_level', axis=1)

df_famd


# In[61]:


from prince import FAMD
from sklearn_extra.cluster import KMedoids
import pandas as pd

# Assuming you have already loaded your dataset into a DataFrame `df_famd`
# df_famd = pd.read_csv('your_data.csv')

# List of categorical columns
categorical_columns = ["gender", "education", "marital", "phq", "gad"]

# Convert each column to 'category' type
for column in categorical_columns:
    df_famd[column] = df_famd[column].astype('category')

# Step 1: Apply FAMD to reduce the dimensionality of mixed data
famd = FAMD(n_components=10, n_iter=3, random_state=42)
famd = famd.fit(df_famd)

# Transform the dataset (projection of the original dataset into the FAMD space)
data_transformed = famd.transform(df_famd)

# Step 2: Apply K-Medoids clustering
kmedoids = KMedoids(n_clusters=3, random_state=42).fit(data_transformed)

# Get the cluster labels for each data point
labels = kmedoids.labels_


# In[62]:


# compute explained variance -> the sum is always 1 by definition
# this means that to understand how many components we have to use, we have to do the scree plot and search for the elbow
eigenvalues = famd.eigenvalues_
explained_variance = eigenvalues / eigenvalues.sum()
print(f'Explained Variance: {explained_variance}')
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.show()


# In[43]:


pip install --upgrade prince


# In[44]:


from sklearn.metrics import silhouette_score

# Assuming `data_transformed` is the DataFrame without the cluster labels
# and `labels` are the cluster labels obtained from K-Medoids

# Calculate the silhouette score
silhouette_avg = silhouette_score(data_transformed, labels)

print(f"The average silhouette_score is: {silhouette_avg}")


# In[45]:


import matplotlib.pyplot as plt

# Assuming `data_transformed` is the transformed DataFrame and `labels` are the cluster labels from K-Medoids

# Plot the clusters
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']

for cluster, color in zip(range(3), colors):
    # Select only data observations with cluster label == current cluster
    cluster_data = data_transformed[labels == cluster]
    # Plot data observations
    plt.scatter(cluster_data[0], cluster_data[1], s=50, color=color, label=f'Cluster {cluster}')

# Label the axes
plt.title('2D visualization of K-Medoids Clusters')
plt.xlabel('FAMD Component 1')
plt.ylabel('FAMD Component 2')

# Show legend
plt.legend()

# Show plot
plt.show()


# In[46]:


df_original = pd.concat([df_sum1, df_numerical], axis=1)


# In[47]:


df_original['Cluster'] = labels


# In[48]:


df_original


# In[49]:


# Define the mapping for education levels
education_mapping = {
    'Elementary': 1,
    'Middle': 2,
    'High': 3,
    'Bachelors': 4,
    'Masters': 5,
    'Doctoral': 6
}

# Apply the mapping to the DataFrame
df_original['education_level_numeric'] = df_original['education_level'].map(education_mapping)

# Verify the transformation
print(df_original[['education_level', 'education_level_numeric']].head())


# In[50]:


df_original = df_original.drop('education_level', axis=1)

df_original


# In[51]:


total_missing_values = df_original.isna().sum().sum()
print(f"Total missing values in the DataFrame: {total_missing_values}")


# In[52]:


df_sum


# In[53]:


# Calculate the variance of each column
variances = df_sum.var()

# Print the variances
print(variances)


# In[54]:


#Kruskal-Wallis Test
feature_columns = [col for col in df_original.columns if col != "Cluster"]
for feature in feature_columns:
    groups = [df_original[df_original['Cluster'] == cluster][[feature]] for cluster in df_original['Cluster'].unique()]
    stat, p = stats.kruskal(*groups)
    print(f"Kruskal-Wallis Test for {feature}:")
    print("Kruskal-Wallis H-statistic:", stat)
    print("p-value:", p)


# In[55]:


# Mann-Whitney U test
feature_names = []
cluster1_values = []
cluster2_values = []
original_p_values = []
corrected_p_values = []
reject_hypothesis = []
for feature in feature_columns:
    clusters = df_original['Cluster'].unique()
    for i, cluster1 in enumerate(clusters):
        for j, cluster2 in enumerate(clusters):
            if i >= j:
                # Skip comparisons of a cluster with itself and duplicate comparisons
                continue

            group1 = df_original[df_original['Cluster'] == cluster1][feature]
            group2 = df_original[df_original['Cluster'] == cluster2][feature]
            stat, p = stats.mannwhitneyu(group1, group2)
            # Bonferroni correction
            alpha = 0.05
            reject, corrected_p_value, _, _ = multipletests(p, alpha=alpha, method='bonferroni')
            feature_names.append(feature)
            cluster1_values.append(cluster1)
            cluster2_values.append(cluster2)
            original_p_values.append(p)
            corrected_p_values.append(corrected_p_value[0])
            reject_hypothesis.append(reject[0])
# Create a DataFrame to display results
results_df = pd.DataFrame({
    'Feature': feature_names,
    'Cluster1': cluster1_values,
    'Cluster2': cluster2_values,
    'Original p-values': original_p_values,
    'Corrected p-values (Bonferroni)': corrected_p_values,
    'Reject Null Hypothesis': reject_hypothesis
})
print(results_df)



# In[56]:


from scipy.stats import fisher_exact, chi2_contingency
# the df for the 2 test is with the columns: gender, marital, education + cluster labels
print(df_sum1)
df_test_categorical = df_sum1[['gender', 'education', 'marital']].copy()
df_test_categorical['Cluster'] = labels

# Create a contingency table
columns_to_test = ['gender', 'education', 'marital']
results = []
for column in columns_to_test:
    contingency_table = pd.crosstab(df_test_categorical[column], df_test_categorical['Cluster'])
    # Print the contingency table and its shape
    print(f"\nContingency Table for {column}:")
    print(contingency_table)
    print(f"Shape: {contingency_table.shape}")

    # Fisher's exact test is only performed on 2x2 tables (variables with 2 categories), so we don't do it

    # Chi-square test
    chi2_stat, chi2_p_value, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-square Test for {column}: Statistic - {chi2_stat}, P-Value - {chi2_p_value}")

    results.append({
        'Variable': column,
        'Chi2_Statistic': chi2_stat,
        'Chi2_P_Value': chi2_p_value
    })


# In[57]:


# Bonferroni correction
alpha = 0.05
correction_method = 'bonferroni'
p_values = [result['Chi2_P_Value'] for result in results]
reject, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method=correction_method)
# Update the results with corrected p-values
for i, result in enumerate(results):
    result['Chi2_Corrected_P_Value'] = corrected_p_values[i]
# Print the results with corrected p-values
for result in results:
    print(f"\nResults for {result['Variable']}:")
    print(f"Chi-square Test - Statistic: {result['Chi2_Statistic']}, Corrected P-Value: {result['Chi2_Corrected_P_Value']}")


# In[58]:


# the df for the 2 test is with the columns: income, age, phq and gad (before making them boolean, eheals, heas, ccs + cluster labels
df_test_numerical = pd.concat([df2, df[['income', 'age']]], axis=1)
df_test_numerical['Cluster'] = labels

#Kruskal-Wallis Test for df_numerical
feature_numerical = [col for col in df_test_numerical.columns if col != "Cluster"]
for feature in feature_numerical:
    groups = [df_test_numerical[df_test_numerical['Cluster'] == cluster][[feature]] for cluster in df_test_numerical['Cluster'].unique()]
    stat, p = stats.kruskal(*groups)
    print(f"Kruskal-Wallis Test for {feature}:")
    print("Kruskal-Wallis H-statistic:", stat)
    print("p-value:", p)

from statsmodels.stats.multitest import multipletests
# Mann-Whitney U test
feature_names = []
cluster1_values = []
cluster2_values = []
original_p_values = []
corrected_p_values = []
reject_hypothesis = []
for feature in feature_numerical:
    clusters = df_test_numerical['Cluster'].unique()
    for i, cluster1 in enumerate(clusters):
        for j, cluster2 in enumerate(clusters):
            if i >= j:
                # Skip comparisons of a cluster with itself and duplicate comparisons
                continue

            group1 = df_test_numerical[df_test_numerical['Cluster'] == cluster1][feature]
            group2 = df_test_numerical[df_test_numerical['Cluster'] == cluster2][feature]
            group1 = np.array(group1)
            group2 = np.array(group2)
            stat, p = stats.mannwhitneyu(group1, group2)
            # Bonferroni correction
            alpha = 0.05
            reject, corrected_p_value, _, _ = multipletests(p, alpha=alpha, method='bonferroni')
            feature_names.append(feature)
            cluster1_values.append(cluster1)
            cluster2_values.append(cluster2)
            original_p_values.append(p)
            corrected_p_values.append(corrected_p_value[0])
            reject_hypothesis.append(reject[0])
# Create a DataFrame to display results
results_df = pd.DataFrame({
    'Feature': feature_names,
    'Cluster1': cluster1_values,
    'Cluster2': cluster2_values,
    'Original p-values': original_p_values,
    'Corrected p-values (Bonferroni)': corrected_p_values,
    'Reject Null Hypothesis': reject_hypothesis
})
print(results_df)


# In[ ]:




