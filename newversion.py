import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency

df = pd.read_csv('../project_data/dataset_project_eHealth20232024.csv')
print(df.info)
print(f'nan in the df: {df.isnull().sum().sum()}')
print(f'rows with at least 1 nan: {df.isnull().T.any().T.sum()}')
print(f'columns with at least 1 nan: \n{df.isnull().any()}')

print(df.median())
df.fillna(value=df.median(), inplace=True)

df.drop_duplicates(keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.info)

col_phq = ['phq_1', 'phq_2', 'phq_3', 'phq_4', 'phq_5', 'phq_6', 'phq_7', 'phq_8', 'phq_9']
sum_phq = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_phq].sum()
    sum_phq.append(row_sum)
print(f'phq sum: {sum_phq}')
# threshold is 10 (over=depression)

col_gad = ['gad_1', 'gad_2', 'gad_3', 'gad_4', 'gad_5', 'gad_6', 'gad_7']
sum_gad = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_gad].sum()
    sum_gad.append(row_sum)
print(f'gad sum: {sum_gad}')
# threshold is 10 (over=anxious)

col_eheals = ['eheals_1', 'eheals_2', 'eheals_3', 'eheals_4', 'eheals_5', 'eheals_6', 'eheals_7', 'eheals_8']
sum_eheals = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_eheals].sum()
    sum_eheals.append(row_sum)
print(f'eheals sum: {sum_eheals}')
# no threshold, its subjective

col_heas = ['heas_1', 'heas_2', 'heas_3', 'heas_4', 'heas_5', 'heas_6', 'heas_7', 'heas_8', 'heas_9', 'heas_10', 'heas_11', 'heas_12', 'heas_13']
sum_heas = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_heas].sum()
    sum_heas.append(row_sum)
print(f'heas sum: {sum_heas}')
# we have to find the threshold!

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

# creating the new dataframe
df1 = df[['age', 'gender', 'education', 'marital', 'income']]
data2 = {
    'phq': sum_phq,
    'gad': sum_gad,
    'eheals': sum_eheals,
    'heas': sum_heas,
    'ccs': sum_ccs
}
df2 = pd.DataFrame(data2)
df_sum = pd.concat([df1, df2], axis=1)
print(df_sum.info)

plt.figure(1)
plt.hist(df_sum['age'],
         bins=np.arange(start=min(df_sum['age']), stop=max(df_sum['age'])+1, step=5),  # distribute the bars every n(=step) years
         color='skyblue', ec='blue')
plt.xlabel('age')
plt.xticks(ticks=np.arange(min(df_sum['age']), max(df_sum['age'])+1, 5))  # adapt tick frequency on x axis
plt.ylabel('n of people')
plt.title('age')

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

plt.figure(5)
plt.hist(df_sum['income'],
         bins=np.arange(start=min(df_sum['income']), stop=max(df_sum['income'])+5000, step=5000),
         color='skyblue', ec='blue')
plt.xlabel('income')
plt.xticks(ticks=np.arange(min(df_sum['income']), max(df_sum['income'])+5000, 5000))
plt.ylabel('n of people')
plt.title('income')

plt.figure(6)
plt.hist(df_sum['income'],
         bins=np.arange(min(df_sum['income']), max(df_sum['income'])+15000, 15000),
         color='skyblue', ec='blue')
plt.xlabel('income')
plt.xticks(ticks=np.arange(min(df_sum['income']), max(df_sum['income'])+15000, 15000))
plt.ylabel('n of people')
plt.title('income')

plt.figure(7)
plt.hist(df_sum['phq'],
         bins=np.arange(start=min(df_sum['phq'])-0.5, stop=max(df_sum['phq'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('phq')
plt.xticks(ticks=np.arange(min(df_sum['phq']), max(df_sum['phq'])+1, 1))
plt.ylabel('n of people')
plt.title('phq score')

plt.figure(8)
plt.hist(df_sum['gad'],
         bins=np.arange(start=min(df_sum['gad'])-0.5, stop=max(df_sum['gad'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('gad')
plt.xticks(ticks=np.arange(min(df_sum['gad']), max(df_sum['gad'])+1, 1))
plt.ylabel('n of people')
plt.title('gad score')

plt.figure(9)
plt.hist(df_sum['eheals'],
         bins=np.arange(start=min(df_sum['eheals'])-0.5, stop=max(df_sum['eheals'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('eheals')
plt.xticks(ticks=np.arange(min(df_sum['eheals']), max(df_sum['eheals'])+1, 2))
plt.ylabel('n of people')
plt.title('eheals score')

plt.figure(10)
plt.hist(df_sum['heas'],
         bins=np.arange(start=min(df_sum['heas'])-0.5, stop=max(df_sum['heas'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('heas')
plt.xticks(ticks=np.arange(min(df_sum['heas']), max(df_sum['heas'])+1, 2))
plt.ylabel('n of people')
plt.title('heas score')

plt.figure(11)
plt.hist(df_sum['ccs'],
         bins=np.arange(start=min(df_sum['ccs'])-0.5, stop=max(df_sum['ccs'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('ccs')
plt.xticks(ticks=np.arange(min(df_sum['ccs']), max(df_sum['ccs'])+1, 3))
plt.ylabel('n of people')
plt.title('ccs score')


# study correlation
plt.figure(12)
sns.heatmap(df_sum.corr(), vmin=-1, vmax=1, center=0, cmap='Spectral', annot=True)
plt.show()

df_numerical = df_sum.drop(columns=['gender', 'education', 'marital'])
df_categorical = df_sum[['gender', 'education', 'marital']].copy()
print(df_numerical)
print(df_categorical)

encoder = OneHotEncoder(sparse_output=False)
categorical_columns = ['gender', 'education', 'marital']  # replace with your actual categorical columns
categorical_encoded = encoder.fit_transform(df_sum[categorical_columns])
categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# scale data
df_scaled_num = RobustScaler().fit_transform(df_numerical)
df_scaled_cat = RobustScaler().fit_transform(categorical_encoded_df)

df_scaled_num = pd.DataFrame(df_scaled_num, columns=['age', 'income', 'phq', 'gad', 'eheals', 'heas', 'ccs'])
df_scaled_cat = pd.DataFrame(df_scaled_cat, columns=encoder.get_feature_names_out(categorical_columns))

# outliers
df_no_outliers = df_scaled_num.copy()
columns_to_process = ['income']
for column in columns_to_process:
    z_scores = np.abs(stats.zscore(df_no_outliers[column]))
    outliers = (z_scores > 1.5)
    median_value = df_no_outliers[column].median()  # Calculate median of the column
    df_no_outliers.loc[outliers, column] = median_value  # Replace outliers with median value
print(df_no_outliers)


df_all = pd.concat([df_scaled_cat, df_no_outliers], axis=1)
print(df_all)


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


# perform pca to find out optimal number of components
pca = PCA()
pca.fit(df_all)
variance = pca.explained_variance_ratio_.cumsum()
plt.figure(12)
print(variance)
plt.plot(range(1, len(variance) + 1), variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance Plot')
plt.show()

# try 7 components
pca = PCA(n_components=7)
df_transformed = pca.fit_transform(df_all)
df_transformed = pd.DataFrame(df_transformed, columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7'])

# Create a KMedoids instance with the number of clusters (K)
kmedoids = KMedoids(n_clusters=3, random_state=0)
kmedoids.fit(df_transformed)
labels = kmedoids.labels_  # contain the cluster assignment for each data point
medoid_indices = kmedoids.medoid_indices_  # contain the indices of medoids in dataset

# plot clusters

# silhouette
silhouette_avg = silhouette_score(df_transformed, labels)
print(f"Silhouette Score: {silhouette_avg}")
print(np.bincount(labels))


feature_columns = [col for col in df_sum.columns]
# Add the cluster labels to the DataFrame
df_sum['Cluster'] = labels


df_numerical['Cluster'] = labels
print(df_numerical)

#Kruskal-Wallis Test for df_numerical
feature_numerical = [col for col in df_numerical.columns if col != "Cluster"]
for feature in feature_numerical:
    groups = [df_numerical[df_numerical['Cluster'] == cluster][[feature]] for cluster in df_numerical['Cluster'].unique()]
    stat, p = stats.kruskal(*groups)
    print(f"Kruskal-Wallis Test for {feature}:")
    print("Kruskal-Wallis H-statistic:", stat)
    print("p-value:", p)

# Mann-Whitney U test
feature_names = []
cluster1_values = []
cluster2_values = []
original_p_values = []
corrected_p_values = []
reject_hypothesis = []
for feature in feature_numerical:
    clusters = df_numerical['Cluster'].unique()
    for i, cluster1 in enumerate(clusters):
        for j, cluster2 in enumerate(clusters):
            if i >= j:
                # Skip comparisons of a cluster with itself and duplicate comparisons
                continue

            group1 = df_numerical[df_numerical['Cluster'] == cluster1][feature]
            group2 = df_numerical[df_numerical['Cluster'] == cluster2][feature]
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


df_categorical['Cluster'] = labels
print(df_categorical)

# Create a contingency table
columns_to_test = ['gender', 'education', 'marital']
results = []
for column in columns_to_test:
    contingency_table = pd.crosstab(df_categorical[column], df_categorical['Cluster'])
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
