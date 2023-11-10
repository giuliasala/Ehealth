#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install prince')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import davies_bouldin_score
from scipy import stats
from prince import MCA


# In[3]:


df=pd.read_csv("/Users/mohamedshoala/Documents/third semester/e health methods and applications/project/project data/dataset_project_eHealth20232024.csv")
print(df.info)
print(f'nan in the df: {df.isnull().sum().sum()}')
print(f'rows with at least 1 nan: {df.isnull().T.any().T.sum()}')
print(f'columns with at least 1 nan: \n{df.isnull().any()}')



# In[4]:


# For other columns, replace missing values with the respective column's median
for column in df.columns:
    median_value = df[column].median()
    df[column].fillna(value=median_value, inplace=True)



# In[5]:


print(f'nan in the df: {df.isnull().sum().sum()}')


# In[6]:


df.drop_duplicates(keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.info)



# In[7]:


col_phq = ['phq_1', 'phq_2', 'phq_3', 'phq_4', 'phq_5', 'phq_6', 'phq_7', 'phq_8', 'phq_9']
sum_phq = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_phq].sum()
    sum_phq.append(row_sum)
print(f'phq sum: {sum_phq}')
# threshold is 10 (over=depression)



# In[8]:


col_gad = ['gad_1', 'gad_2', 'gad_3', 'gad_4', 'gad_5', 'gad_6', 'gad_7']
sum_gad = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_gad].sum()
    sum_gad.append(row_sum)
print(f'gad sum: {sum_gad}')
# threshold is 10 (over=anxious)



# In[9]:


col_eheals = ['eheals_1', 'eheals_2', 'eheals_3', 'eheals_4', 'eheals_5', 'eheals_6', 'eheals_7', 'eheals_8']
sum_eheals = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_eheals].sum()
    sum_eheals.append(row_sum)
print(f'eheals sum: {sum_eheals}')
# no threshold, its subjective



# In[10]:


col_heas = ['heas_1', 'heas_2', 'heas_3', 'heas_4', 'heas_5', 'heas_6', 'heas_7', 'heas_8', 'heas_9', 'heas_10', 'heas_11', 'heas_12', 'heas_13']
sum_heas = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_heas].sum()
    sum_heas.append(row_sum)
print(f'heas sum: {sum_heas}')
# we have to find the threshold!



# In[11]:


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



# In[12]:


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



# In[13]:


df_numerical=df[["age","income"]]
df_numerical


# In[14]:


plt.figure(2)
plt.hist(df_sum['gender'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('gender')
labels_gen = ['male', 'female', 'non binary', 'prefer not to say']
plt.xticks(ticks=[0, 1, 2, 3], labels=labels_gen)
plt.ylabel('n of people')
plt.title('gender')

plt.figure(3)
bin_centers = [np.mean([bin_left, bin_right]) for bin_left, bin_right in zip([0, 5, 8, 13, 18, 22, 25], [5, 8, 13, 18, 22, 25, 28])]
plt.hist(df_sum['education'], bins=bin_centers, color='skyblue', ec='blue')
plt.xlabel('education')
labels_edu = ['elem.', 'middle', 'high', 'bachelor', 'master', 'doctoral']
plt.xticks(ticks=[5, 8, 13, 18, 22, 25], labels=labels_edu)
plt.ylabel('n of people')
plt.title('education')

plt.figure(4)
plt.hist(df_sum['marital'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color='skyblue', ec='blue')
plt.xlabel('marital status')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=['single', 'married', 'divorced', 'widowed', 'separated', 'pnts'])
plt.ylabel('n of people')
plt.title('marital status')



plt.figure(6)
plt.hist(df_sum['phq'],
         bins=np.arange(start=min(df_sum['phq'])-0.5, stop=max(df_sum['phq'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('phq')
plt.xticks(ticks=np.arange(min(df_sum['phq']), max(df_sum['phq'])+1, 1))
plt.ylabel('n of people')
plt.title('phq score')

plt.figure(7)
plt.hist(df_sum['gad'],
         bins=np.arange(start=min(df_sum['gad'])-0.5, stop=max(df_sum['gad'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('gad')
plt.xticks(ticks=np.arange(min(df_sum['gad']), max(df_sum['gad'])+1, 1))
plt.ylabel('n of people')
plt.title('gad score')

plt.figure(8)
plt.hist(df_sum['eheals'],
         bins=np.arange(start=min(df_sum['eheals'])-0.5, stop=max(df_sum['eheals'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('eheals')
plt.xticks(ticks=np.arange(min(df_sum['eheals']), max(df_sum['eheals'])+1, 2))
plt.ylabel('n of people')
plt.title('eheals score')

plt.figure(9)
plt.hist(df_sum['heas'],
         bins=np.arange(start=min(df_sum['heas'])-0.5, stop=max(df_sum['heas'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('heas')
plt.xticks(ticks=np.arange(min(df_sum['heas']), max(df_sum['heas'])+1, 2))
plt.ylabel('n of people')
plt.title('heas score')

plt.figure(10)
plt.hist(df_sum['ccs'],
         bins=np.arange(start=min(df_sum['ccs'])-0.5, stop=max(df_sum['ccs'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('ccs')
plt.xticks(ticks=np.arange(min(df_sum['ccs']), max(df_sum['ccs'])+1, 3))
plt.ylabel('n of people')
plt.title('ccs score')

# study correlation
plt.figure(11)
sns.heatmap(df_sum.corr(), vmin=-1, vmax=1, center=0, cmap='Spectral', annot=True)
plt.show()


# In[15]:


from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler instance
scaler = MinMaxScaler()

# Scale all columns in your DataFrame (let's call it df)
df_scaled = pd.DataFrame(scaler.fit_transform(df_sum), columns=df_sum.columns)

# Now, df_scaled contains the scaled values for all columns in the DataFrame


# In[16]:


plt.figure(2)
plt.hist(df_scaled['gender'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('gender')
labels_gen = ['male', 'female', 'non binary', 'prefer not to say']
plt.xticks(ticks=[0, 1, 2, 3], labels=labels_gen)
plt.ylabel('n of people')
plt.title('gender')

plt.figure(3)
bin_centers = [np.mean([bin_left, bin_right]) for bin_left, bin_right in zip([0, 5, 8, 13, 18, 22, 25], [5, 8, 13, 18, 22, 25, 28])]
plt.hist(df_scaled['education'], bins=bin_centers, color='skyblue', ec='blue')
plt.xlabel('education')
labels_edu = ['elem.', 'middle', 'high', 'bachelor', 'master', 'doctoral']
plt.xticks(ticks=[5, 8, 13, 18, 22, 25], labels=labels_edu)
plt.ylabel('n of people')
plt.title('education')

plt.figure(4)
plt.hist(df_scaled['marital'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color='skyblue', ec='blue')
plt.xlabel('marital status')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=['single', 'married', 'divorced', 'widowed', 'separated', 'pnts'])
plt.ylabel('n of people')
plt.title('marital status')



plt.figure(6)
plt.hist(df_scaled['phq'],
         bins=np.arange(start=min(df_scaled['phq'])-0.5, stop=max(df_scaled['phq'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('phq')
plt.xticks(ticks=np.arange(min(df_scaled['phq']), max(df_scaled['phq'])+1, 1))
plt.ylabel('n of people')
plt.title('phq score')

plt.figure(7)
plt.hist(df_scaled['gad'],
         bins=np.arange(start=min(df_scaled['gad'])-0.5, stop=max(df_scaled['gad'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('gad')
plt.xticks(ticks=np.arange(min(df_scaled['gad']), max(df_scaled['gad'])+1, 1))
plt.ylabel('n of people')
plt.title('gad score')

plt.figure(8)
plt.hist(df_scaled['eheals'],
         bins=np.arange(start=min(df_scaled['eheals'])-0.5, stop=max(df_scaled['eheals'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('eheals')
plt.xticks(ticks=np.arange(min(df_scaled['eheals']), max(df_scaled['eheals'])+1, 2))
plt.ylabel('n of people')
plt.title('eheals score')

plt.figure(9)
plt.hist(df_scaled['heas'],
         bins=np.arange(start=min(df_scaled['heas'])-0.5, stop=max(df_scaled['heas'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('heas')
plt.xticks(ticks=np.arange(min(df_scaled['heas']), max(df_scaled['heas'])+1, 2))
plt.ylabel('n of people')
plt.title('heas score')

plt.figure(10)
plt.hist(df_scaled['ccs'],
         bins=np.arange(start=min(df_scaled['ccs'])-0.5, stop=max(df_scaled['ccs'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('ccs')
plt.xticks(ticks=np.arange(min(df_scaled['ccs']), max(df_scaled['ccs'])+1, 3))
plt.ylabel('n of people')
plt.title('ccs score')

# study correlation
plt.figure(11)
sns.heatmap(df_scaled.corr(), vmin=-1, vmax=1, center=0, cmap='Spectral', annot=True)
plt.show()


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns

# Define the bin edges for each plot
bin_edges_gender = [-0.5, 0.5, 1.5, 2.5, 3.5]
bin_centers_education = [0, 1, 2, 3, 4, 5]
bin_centers_marital = [0, 1, 2, 3, 4, 5]

# Adjust the bin distances based on the new min-max scaled data
plt.figure(2)
plt.hist(df_scaled['gender'], bins=bin_edges_gender, color='skyblue', ec='blue')
plt.xlabel('gender')
labels_gen = ['male', 'female', 'non binary', 'prefer not to say']
plt.xticks(ticks=[0, 1, 2, 3], labels=labels_gen)
plt.ylabel('n of people')
plt.title('gender')

plt.figure(3)
plt.hist(df_scaled['education'], bins=bin_centers_education, color='skyblue', ec='blue')
plt.xlabel('education')
labels_edu = ['elem.', 'middle', 'high', 'bachelor', 'master', 'doctoral']
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=labels_edu)
plt.ylabel('n of people')
plt.title('education')

plt.figure(4)
plt.hist(df_scaled['marital'], bins=bin_centers_marital, color='skyblue', ec='blue')
plt.xlabel('marital status')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=['single', 'married', 'divorced', 'widowed', 'separated', 'pnts'])
plt.ylabel('n of people')
plt.title('marital status')

# Adjust bin distances for the remaining plots (6 to 11) in a similar manner

# study correlation
plt.figure(12)
sns.heatmap(df_scaled.corr(), vmin=-1, vmax=1, center=0, cmap='Spectral', annot=True)
plt.show()


# In[18]:


#BOX_PLOT
get_ipython().run_line_magic('matplotlib', 'inline')
df_scaled.boxplot(rot=90)


# In[19]:


#BOX_PLOT
get_ipython().run_line_magic('matplotlib', 'inline')
df_sum.boxplot(rot=90)


# In[20]:


sns.pairplot(df_sum)
plt.show()


# In[21]:


sns.pairplot(df_scaled)
plt.show()


# In[22]:


#so after scaling with min_max scaler we still have the same representation of all the categories 


# In[23]:


scaler = StandardScaler()
df_standardize = scaler.fit_transform(df_numerical)

# Convert the standardized NumPy array back to a Pandas DataFrame
df_standardize = pd.DataFrame(df_standardize, columns=df_numerical.columns)


# In[24]:


df_numerical


# In[25]:


#BOX_PLOT
get_ipython().run_line_magic('matplotlib', 'inline')
df_numerical.boxplot(rot=90)


# In[26]:


df_standardize


# In[27]:


#BOX_PLOT
get_ipython().run_line_magic('matplotlib', 'inline')
df_standardize.boxplot(rot=90)


# In[28]:


# Calculate the medians for each column in the original DataFrame
medians = df_standardize.median()

# List of columns you want to process
columns_to_process = ['income']  # Add the names of the columns you want to process
threshold = 1.5

# Create a copy of the DataFrame without outliers
df_no_outliers = df_standardize.copy()

for column in columns_to_process:
    z_scores = np.abs(stats.zscore(df_no_outliers[column]))
    outliers = (z_scores > threshold)

    # Replace outliers with the median value of the column
    df_no_outliers[column][outliers] = medians[column]


# In[29]:


df_no_outliers


# In[30]:


#BOX_PLOT
get_ipython().run_line_magic('matplotlib', 'inline')
df_no_outliers.boxplot(rot=90)


# In[31]:


if df_no_outliers.isna().values.any():
    print("There are NaN values in the DataFrame.")
else:
    print("No NaN values in the DataFrame.")


# In[32]:


df_all = pd.concat([df_scaled, df_no_outliers], axis=1)
df_all


# In[33]:


#BOX_PLOT
get_ipython().run_line_magic('matplotlib', 'inline')
df_all.boxplot(rot=90)


# In[36]:


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


# In[ ]:


plt.plot(k_values, wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.title("Elbow Method for Optimal K")
plt.show()


# In[37]:


# Specify the number of clusters (K) and the maximum number of iterations (max_iter)
k = 3
max_iter = 100

# Create a KMedoids instance and fit the data
kmedoids = KMedoids(n_clusters=k, max_iter=max_iter, random_state=0)
kmedoids.fit(df_all)

# Get the cluster assignments
cluster_labels = kmedoids.labels_

# Add the cluster labels to the DataFrame
df_all['Cluster'] = cluster_labels


# In[38]:


silhouette_avg = silhouette_score(df_all, cluster_labels)

print(f"Silhouette Score: {silhouette_avg}")


# In[ ]:


# Calculate the medians for each column in the original DataFrame
medians = df_standardize.median()

# List of columns you want to process
columns_to_process = ['income']  # Add the names of the columns you want to process
threshold = 1.5

# Create a copy of the DataFrame without outliers
df_no_outliers = df_standardize.copy()

for column in columns_to_process:
    z_scores = np.abs(stats.zscore(df_no_outliers[column]))
    outliers = (z_scores > threshold)

    # Replace outliers with the median value of the column
    df_no_outliers[column][outliers] = medians[column]


# In[ ]:


# Install the 'mca' library if you haven't already
# pip install mca



# Assuming you have already performed MCA and stored the results in 'mca_results'
# For example:
# mca = MCA(n_components=3)  # You can choose the number of components you need
# mca.fit(df_categorical)    # 'df_categorical' is your categorical data
# mca_results = mca.transform(df_categorical)

# Calculate the explained variance for each component


# In[44]:


mca = MCA(n_components=2)  # You can choose the number of components you need
mca.fit(df_scaled)
mca_results = mca.transform(df_scaled)

# Define your desired column names as a list of strings
new_column_names = ["one", "two"]  # Replace these names as needed

# Rename the columns of the DataFrame
mca_results.columns = new_column_names


# In[40]:


mca_results


# In[49]:


# Assuming you have already performed MCA and stored the results in 'mca_results'
# For example:
# mca = MCA(n_components=3)  # You can choose the number of components you need
# mca.fit(df_categorical)    # 'df_categorical' is your categorical data
# mca_results = mca.transform(df_categorical)

# Calculate the singular values (eigenvalues) from MCA results
singular_values = mca.singular_values_

# Calculate the explained variance for each component
explained_var = (singular_values ** 2) / np.sum(singular_values ** 2)

# Calculate the cumulative explained variance
cumulative_var = explained_var.cumsum()

# Print or visualize the explained variance and cumulative explained variance
print("Explained Variance for Each Component:")
print(explained_var)
print("\nCumulative Explained Variance:")
print(cumulative_var)

# You can create a scree plot to visualize the explained variance
import matplotlib.pyplot as plt

plt.plot(range(1, len(explained_var) + 1), cumulative_var, marker='o', linestyle='--')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Scree Plot for MCA")
plt.grid()
plt.show()


# In[46]:


eigenvalues = mca.eigenvalues_

# Calculate the total inertia
total_inertia = sum(eigenvalues)

# Calculate the explained variance for each component
explained_var_ratio = eigenvalues / total_inertia

# Print or analyze the explained variance ratio for each component
for i, ratio in enumerate(explained_var_ratio):
    print(f"Component {i + 1}: Explained Variance Ratio = {ratio:.2f}")


# In[42]:


df_all = pd.concat([mca_results, df_no_outliers], axis=1)
df_all


# In[ ]:


k_values = range(1, 11)
wcss = []  # Within-Cluster Sum of Squares

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(df_all)
    wcss.append(kmeans.inertia_)


# In[ ]:


plt.plot(k_values, wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.title("Elbow Method for Optimal K")
plt.show()


# In[50]:


# Specify the number of clusters (K) and the maximum number of iterations (max_iter)
k = 3
max_iter = 100

# Create a KMedoids instance and fit the data
kmedoids = KMedoids(n_clusters=k, max_iter=max_iter, random_state=0)
kmedoids.fit(df_all)

# Get the cluster assignments
cluster_labels = kmedoids.labels_

# Add the cluster labels to the DataFrame
df_all['Cluster'] = cluster_labels


# In[51]:


silhouette_avg = silhouette_score(df_all, cluster_labels)

print(f"Silhouette Score: {silhouette_avg}")


# In[52]:


df_all


# In[67]:


df_original = pd.concat([df_sum, df_numerical], axis=1)


# In[68]:


df_original['Cluster'] = cluster_labels


# In[69]:


df_original


# In[60]:


df_sum


# In[75]:


#Kruskal-Wallis Test
feature_columns = [col for col in df_original.columns if col != "Cluster"]
for feature in feature_columns:
    groups = [df_original[df_original['Cluster'] == cluster][[feature]] for cluster in df_original['Cluster'].unique()]
    stat, p = stats.kruskal(*groups)
    print(f"Kruskal-Wallis Test for {feature}:")
    print("Kruskal-Wallis H-statistic:", stat)
    print("p-value:", p)


# In[76]:


from statsmodels.stats.multitest import multipletests
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


# In[ ]:


sns.pairplot(df_no_outliers)
plt.show()


# In[ ]:




