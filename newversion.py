# import all the libraries that we need
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from scipy.stats import kstest
from prince import FAMD

# read the cvs file
df = pd.read_csv('../project_data/dataset_project_eHealth20232024.csv')
# print information about the dataframe
print(df.info)

# print the number NaN in the dataframe, the number of rows and the columns that contain NaN
print(f'nan in the df: {df.isnull().sum().sum()}')
print(f'rows with at least 1 nan: {df.isnull().T.any().T.sum()}')
print(f'columns with at least 1 nan: \n{df.isnull().any()}')
# replace the NaN with the median of the column
print(df.median())
df.fillna(value=df.median(), inplace=True)

# drop the duplicate rows in the dataframe
df.drop_duplicates(keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.info)

# create one attribute for each questionnaire, by summing all the scores of the questions
col_phq = ['phq_1', 'phq_2', 'phq_3', 'phq_4', 'phq_5', 'phq_6', 'phq_7', 'phq_8', 'phq_9']
sum_phq = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_phq].sum()
    sum_phq.append(row_sum)
print(f'phq sum: {sum_phq}')

col_gad = ['gad_1', 'gad_2', 'gad_3', 'gad_4', 'gad_5', 'gad_6', 'gad_7']
sum_gad = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_gad].sum()
    sum_gad.append(row_sum)
print(f'gad sum: {sum_gad}')

col_eheals = ['eheals_1', 'eheals_2', 'eheals_3', 'eheals_4', 'eheals_5', 'eheals_6', 'eheals_7', 'eheals_8']
sum_eheals = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_eheals].sum()
    sum_eheals.append(row_sum)
print(f'eheals sum: {sum_eheals}')

col_heas = ['heas_1', 'heas_2', 'heas_3', 'heas_4', 'heas_5', 'heas_6', 'heas_7', 'heas_8', 'heas_9', 'heas_10', 'heas_11', 'heas_12', 'heas_13']
sum_heas = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_heas].sum()
    sum_heas.append(row_sum)
print(f'heas sum: {sum_heas}')

# the higher the score, the more skeptic the person is about climate change
# except for the following questions
# for this reason, we reverse the scores of ccs 3, 6, 7 and 12 before summing
max_score = 6
df['ccs_3'] = max_score - df['ccs_3']
df['ccs_6'] = max_score - df['ccs_6']
df['ccs_7'] = max_score - df['ccs_7']
df['ccs_12'] = max_score - df['ccs_12']

col_ccs = ['ccs_1', 'ccs_2', 'ccs_3', 'ccs_4', 'ccs_5', 'ccs_6', 'ccs_7', 'ccs_8', 'ccs_9', 'ccs_10', 'ccs_11', 'ccs_12']
sum_ccs = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_ccs].sum()
    sum_ccs.append(row_sum)
print(f'ccs sum: {sum_ccs}')

# create the new dataframe (df_sum)
# with the original columns for age, gender, education, marital and income
# and with the new columns for the questionnaires
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
print('\ndf_sum:')
print(df_sum.info)

# create two different dataframes for categorical (gender, education, marital)
# and numerical variables (age, income and the 5 questionnaires: phq, gad, eheals, heas, ccs)
df_numerical = df_sum.drop(columns=['gender', 'education', 'marital'])
df_categorical = df_sum[['gender', 'education', 'marital']].copy()
print('\ndf_numerical:')
print(df_numerical)
print('\ndf_categorical:')
print(df_categorical)

# boxplot of the numerical variables, used to see if there are any outliers
attributes_to_plot = [col for col in df_numerical.columns]
plt.figure(figsize=(10, 6 * len(attributes_to_plot)))
for i, attribute in enumerate(attributes_to_plot, 1):
    plt.subplot(len(attributes_to_plot), 1, i)
    plt.boxplot(df_numerical[attribute])
    plt.title(f'Boxplot of {attribute}')
    plt.ylabel('Values')
    plt.xlabel(attribute)
    plt.tight_layout()

# we can see that only income has outliers and they are
# identify outliers using the Inter-Quartile Range method
df_no_outliers = df_numerical.copy()
Q1 = df_no_outliers['income'].quantile(0.25)
Q3 = df_no_outliers['income'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = ((df_no_outliers['income'] < lower_bound) | (df_no_outliers['income'] > upper_bound))
print('\nRows with outliers:')
print(df_no_outliers[outliers])
# Winsorizing: cap extreme values by replacing them with the upper bound
df_no_outliers['income'] = df_no_outliers['income'].apply(lambda x: upper_bound if x > upper_bound else x)  # Replace outliers with the highest value in the range
print(df_no_outliers.iloc[[28, 52]])

# plot histograms for each variable
plt.figure()
plt.hist(df_sum['age'],
         bins=np.arange(start=min(df_sum['age']), stop=max(df_sum['age'])+1, step=5),  # distribute the bars every n(=step) years
         color='skyblue', ec='blue')
plt.xlabel('age')
plt.xticks(ticks=np.arange(min(df_sum['age']), max(df_sum['age'])+1, 5))  # adapt tick frequency on x axis
plt.ylabel('n of people')
plt.title('age')

plt.figure()
plt.hist(df_sum['gender'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('gender')
labels_gen = ['male', 'female', 'non binary', 'prefer not to say']
plt.xticks(ticks=[0, 1, 2, 3], labels=labels_gen)
plt.ylabel('n of people')
plt.title('gender')

plt.figure()
bin_centers = [np.mean([bin_left, bin_right]) for bin_left, bin_right in zip([0, 5, 8, 13, 18, 22, 25], [5, 8, 13, 18, 22, 25, 28])]
plt.hist(df_sum['education'], bins=bin_centers, color='skyblue', ec='blue')
plt.xlabel('education')
labels_edu = ['elem.', 'middle', 'high', 'bachelor', 'master', 'doctoral']
plt.xticks(ticks=[5, 8, 13, 18, 22, 25], labels=labels_edu)
plt.ylabel('n of people')
plt.title('education')

plt.figure()
plt.hist(df_sum['marital'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color='skyblue', ec='blue')
plt.xlabel('marital status')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=['single', 'married', 'divorced', 'widowed', 'separated', 'pnts'])
plt.ylabel('n of people')
plt.title('marital status')

plt.figure()
plt.hist(df_sum['income'],
         bins=np.arange(start=min(df_sum['income']), stop=max(df_sum['income'])+5000, step=5000),
         color='skyblue', ec='blue')
plt.xlabel('income')
plt.xticks(ticks=np.arange(min(df_sum['income']), max(df_sum['income'])+5000, 5000))
plt.ylabel('n of people')
plt.title('income')

plt.figure()
plt.hist(df_sum['phq'],
         bins=np.arange(start=min(df_sum['phq'])-0.5, stop=max(df_sum['phq'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('phq')
plt.xticks(ticks=np.arange(min(df_sum['phq']), max(df_sum['phq'])+1, 1))
plt.ylabel('n of people')
plt.title('phq score')

plt.figure()
plt.hist(df_sum['gad'],
         bins=np.arange(start=min(df_sum['gad'])-0.5, stop=max(df_sum['gad'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('gad')
plt.xticks(ticks=np.arange(min(df_sum['gad']), max(df_sum['gad'])+1, 1))
plt.ylabel('n of people')
plt.title('gad score')

plt.figure()
plt.hist(df_sum['eheals'],
         bins=np.arange(start=min(df_sum['eheals'])-0.5, stop=max(df_sum['eheals'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('eheals')
plt.xticks(ticks=np.arange(min(df_sum['eheals']), max(df_sum['eheals'])+1, 2))
plt.ylabel('n of people')
plt.title('eheals score')

plt.figure()
plt.hist(df_sum['heas'],
         bins=np.arange(start=min(df_sum['heas'])-0.5, stop=max(df_sum['heas'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('heas')
plt.xticks(ticks=np.arange(min(df_sum['heas']), max(df_sum['heas'])+1, 2))
plt.ylabel('n of people')
plt.title('heas score')

plt.figure()
plt.hist(df_sum['ccs'],
         bins=np.arange(start=min(df_sum['ccs'])-0.5, stop=max(df_sum['ccs'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xlabel('ccs')
plt.xticks(ticks=np.arange(min(df_sum['ccs']), max(df_sum['ccs'])+1, 3))
plt.ylabel('n of people')
plt.title('ccs score')

# print summary statistics for numerical variables
print(df_no_outliers.describe())

#pairplot for numerical variables
sns.pairplot(df_no_outliers)

# study correlation between all the variables by plotting the heatmap
plt.figure(12)
sns.heatmap(df_sum.corr(), vmin=-1, vmax=1, center=0, cmap='Spectral', annot=True)
plt.show()

# we can drop the ccs variable, as we see that it is strongly correlated with all the other questionnaires
df_no_outliers.drop('ccs', axis=1, inplace=True)

# perform Kolmogorov-Smirnov test for normality
for col in df_numerical.columns:
    ks_statistic, ks_p_value = kstest(df_numerical[col], 'norm')
    print(f'Kolmogorov-Smirnov test for {col}:')
    print(f'KS Statistic: {ks_statistic}')
    print(f'p-value: {ks_p_value}')
# we reject the hipothesis that data is normally distributed

# one hot encoding of categorical columns for pca
encoder = OneHotEncoder(sparse_output=False)
categorical_columns = ['gender', 'education', 'marital']
categorical_encoded = encoder.fit_transform(df_sum[categorical_columns])
categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# replacing education values with 0, 1, 2, 3, 4 for famd
education_mapping = {
    5: 0,
    8: 1,
    13: 2,
    18: 3,
    22: 4,
    25: 5
}
df_categorical.loc[:, 'education'] = df_categorical['education'].map(education_mapping)
print(df_categorical)


# df to perform pca
df_tot_pca = pd.concat([categorical_encoded_df, df_no_outliers], axis=1)
print(df_tot_pca)
# scale data
df_all_pca = RobustScaler().fit_transform(df_tot_pca)
df_all_pca = pd.DataFrame(df_all_pca, columns=[col for col in df_tot_pca.columns])
print(df_all_pca)

# df to perform famd
df_tot_famd = pd.concat([df_categorical, df_no_outliers], axis=1)
print(df_tot_famd)
# scale data
df_all_famd = RobustScaler().fit_transform(df_tot_famd)
df_all_famd = pd.DataFrame(df_all_famd, columns=[col for col in df_tot_famd.columns])
print(df_all_famd)

# no preprocessing, using the one hot encoded df
# elbow method
inertia = []  # empty list to store the sum of squared distances for different K values
k_values = range(1, 11)
for k in k_values:
    kmedoids = KMedoids(n_clusters=k, random_state=0)
    kmedoids.fit(df_all_pca)
    inertia.append(kmedoids.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for no pre-processing, one hot encoded')
plt.grid()
plt.show()

# no preprocessing, using the df without one hot encoding
# elbow method
inertia = []  # empty list to store the sum of squared distances for different K values
k_values = range(1, 11)
for k in k_values:
    kmedoids = KMedoids(n_clusters=k, random_state=0)
    kmedoids.fit(df_all_famd)
    inertia.append(kmedoids.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for no pre-processing')
plt.grid()
plt.show()

# perform pca to find out optimal number of components
pca = PCA()
pca.fit(df_all_pca)
variance = pca.explained_variance_ratio_.cumsum()
plt.figure(12)
print(variance)
plt.plot(range(1, len(variance) + 1), variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance Plot')
plt.show()

pca = PCA(n_components=7)
data_transformed_pca = pca.fit_transform(df_all_pca)
df_transformed_pca = pd.DataFrame(data_transformed_pca, columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7'])

# elbow method
inertia = []  # empty list to store the sum of squared distances for different K values
k_values = range(1, 11)
for k in k_values:
    kmedoids = KMedoids(n_clusters=k, random_state=0)
    kmedoids.fit(df_transformed_pca)
    inertia.append(kmedoids.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for PCA')
plt.grid()
plt.show()

# KMedoids
kmedoids = KMedoids(n_clusters=3, random_state=0, init='k-medoids++', metric='seuclidean', method='pam', max_iter=100)
kmedoids.fit_transform(df_transformed_pca)
labels_pca = kmedoids.labels_  # contain the cluster assignment for each data point
medoid_indices_pca = kmedoids.medoid_indices_  # contain the indices of medoids in dataset

# silhouette score
silhouette_avg = silhouette_score(df_transformed_pca, labels_pca)
print(f"Silhouette Score for PCA: {silhouette_avg}")
print(np.bincount(labels_pca))


# Convert each column to 'category' type
categorical_columns = ["gender", "education", "marital"]
for column in categorical_columns:
    df_all_famd[categorical_columns] = df_all_famd[categorical_columns].astype('category')

# perform famd
famd = FAMD(n_components=10, n_iter=7, random_state=42)
famd.fit_transform(df_all_famd)
# compute explained variance -> the sum is always 1 by definition
# this means that to understand how many components we have to use, we have to do the scree plot and search for the elbow
eigenvalues = famd.eigenvalues_
explained_variance = eigenvalues / eigenvalues.sum()
cumulative_variance=np.cumsum(explained_variance)
print(f'Explained Variance: {cumulative_variance}')
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.show()

# we see that the variance expained by the first 6 components is enough
famd = FAMD(n_components=6, n_iter=7, random_state=42)
data_transformed_famd = famd.fit_transform(df_all_famd)
df_transformed_famd = pd.DataFrame(data_transformed_famd)

inertia = []  # empty list to store the sum of squared distances for different K values
k_values = range(1, 11)
for k in k_values:
    kmedoids = KMedoids(n_clusters=k, random_state=0)
    kmedoids.fit(df_transformed_famd)
    inertia.append(kmedoids.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for FAMD')
plt.grid()
plt.show()

# KMedoids
kmedoids = KMedoids(n_clusters=3, random_state=0, init='k-medoids++', metric='seuclidean', method='pam', max_iter=100)
kmedoids.fit(df_transformed_famd)
labels_famd = kmedoids.labels_  # contain the cluster assignment for each data point
medoid_indices_famd = kmedoids.medoid_indices_  # contain the indices of medoids in dataset

# silhouette score
silhouette_avg = silhouette_score(df_transformed_famd, labels_famd)
print(f"Silhouette Score for FAMD: {silhouette_avg}")
print(np.bincount(labels_famd))

# analytical method for the silhouette, still have to implement it!!!!!


feature_columns = [col for col in df_sum.columns]

# Add the cluster labels of ... to the DataFrame (see analytical method above, still need to decide!!!!)
df_numerical['Cluster'] = labels_pca #for now lets see pca
print(df_numerical)

# Kruskal-Wallis Test for df_numerical
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

# same as numerical !!!!
df_categorical['Cluster'] = labels_pca
print(df_categorical)

columns_to_test = [col for col in df_categorical.columns if col != "Cluster"]
for column in columns_to_test:
    contingency_table = pd.crosstab(df_categorical[column], df_categorical['Cluster'])
    print(f"\nContingency Table for {column}:")
    print(contingency_table)
    print(f"Shape: {contingency_table.shape}")
    contingency_table.to_csv(f"{column}_contingency_table.csv", index=True, header=True)

cluster_summary = df_numerical.groupby('Cluster').mean()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(cluster_summary)