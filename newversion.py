# import all the libraries that we need
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn_extra.cluster import KMedoids
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from scipy.stats import kstest
from prince import FAMD
import time

plt.rcParams['figure.max_open_warning'] = 50

# read the cvs file
df = pd.read_csv('../project_data/dataset_project_eHealth20232024.csv')
# print the dataframe
print('\ndataframe:')
print(df)

# CLEANING
# print the number NaN in the dataframe, the number of rows and the columns that contain NaN
print(f'nan in the df: {df.isnull().sum().sum()}')
print(f'rows with at least 1 nan: {df.isnull().T.any().T.sum()}')
print(f'columns with at least 1 nan: \n{df.isnull().any()}')
# replace the NaN with the median of the column
print('\nmedian values:')
print(df.median())
df.fillna(value=df.median(), inplace=True)

# drop the duplicate rows in the dataframe
df.drop_duplicates(keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
print('\ndataframe after dropping duplicates:')
print(df)

# create one attribute for each questionnaire, by summing all the scores of the questions
col_phq = ['phq_1', 'phq_2', 'phq_3', 'phq_4', 'phq_5', 'phq_6', 'phq_7', 'phq_8', 'phq_9']
sum_phq = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_phq].sum()
    sum_phq.append(row_sum)
print(f'PHQ sum: {sum_phq}')

col_gad = ['gad_1', 'gad_2', 'gad_3', 'gad_4', 'gad_5', 'gad_6', 'gad_7']
sum_gad = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_gad].sum()
    sum_gad.append(row_sum)
print(f'GAD sum: {sum_gad}')

col_eheals = ['eheals_1', 'eheals_2', 'eheals_3', 'eheals_4', 'eheals_5', 'eheals_6', 'eheals_7', 'eheals_8']
sum_eheals = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_eheals].sum()
    sum_eheals.append(row_sum)
print(f'EHEALS sum: {sum_eheals}')

col_heas = ['heas_1', 'heas_2', 'heas_3', 'heas_4', 'heas_5', 'heas_6', 'heas_7', 'heas_8', 'heas_9', 'heas_10', 'heas_11', 'heas_12', 'heas_13']
sum_heas = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_heas].sum()
    sum_heas.append(row_sum)
print(f'HEAS sum: {sum_heas}')

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
print(f'CCS sum: {sum_ccs}')

# create the new dataframe (df_sum)
pd.set_option('display.max_columns', None)  # set this option in order to see all columns when printing the dataframes
# take original columns for age, gender, education, marital and income
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
print('\ndf_sum info:')
print(df_sum.info())
print(df_sum)

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

# we can see that only income has outliers
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
print('\nSame rows, values replaced:')
print(df_no_outliers.iloc[[28, 52]])

# EDA
# Uni-variate analysis:
# see also boxplots plotted before

# print summary statistics for numerical variables
print('\nSummary statistics for numerical variables:')
print(df_no_outliers.describe())

# plot histograms for numerical variables
plt.figure()
plt.hist(df_no_outliers['age'],
         bins=np.arange(start=min(df_sum['age']), stop=max(df_sum['age'])+1, step=5),  # distribute the bars every n(=step) years
         color='skyblue', ec='blue')
plt.xticks(ticks=np.arange(min(df_no_outliers['age']), max(df_no_outliers['age'])+1, 5))  # adapt tick frequency on x axis
plt.ylabel('n of people')
plt.title('Histogram of age')

plt.figure()
plt.hist(df_no_outliers['income'],
         bins=np.arange(start=min(df_no_outliers['income']), stop=max(df_no_outliers['income'])+5000, step=5000),
         color='skyblue', ec='blue')
plt.xticks(ticks=np.arange(min(df_no_outliers['income']), max(df_no_outliers['income'])+5000, 5000))
plt.ylabel('n of people')
plt.title('Histogram of income')

plt.figure()
plt.hist(df_no_outliers['phq'],
         bins=np.arange(start=min(df_no_outliers['phq'])-0.5, stop=max(df_no_outliers['phq'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xticks(ticks=np.arange(min(df_no_outliers['phq']), max(df_no_outliers['phq'])+1, 1))
plt.ylabel('n of people')
plt.title('Histogram of PHQ score')

plt.figure()
plt.hist(df_no_outliers['gad'],
         bins=np.arange(start=min(df_no_outliers['gad'])-0.5, stop=max(df_no_outliers['gad'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xticks(ticks=np.arange(min(df_no_outliers['gad']), max(df_no_outliers['gad'])+1, 1))
plt.ylabel('n of people')
plt.title('Histogram of GAD score')

plt.figure()
plt.hist(df_no_outliers['eheals'],
         bins=np.arange(start=min(df_no_outliers['eheals'])-0.5, stop=max(df_no_outliers['eheals'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xticks(ticks=np.arange(min(df_no_outliers['eheals']), max(df_no_outliers['eheals'])+1, 2))
plt.ylabel('n of people')
plt.title('Histogram of EHEALS score')

plt.figure()
plt.hist(df_no_outliers['heas'],
         bins=np.arange(start=min(df_no_outliers['heas'])-0.5, stop=max(df_no_outliers['heas'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xticks(ticks=np.arange(min(df_no_outliers['heas']), max(df_no_outliers['heas'])+1, 2))
plt.ylabel('n of people')
plt.title('Histogram of HEAS score')

plt.figure()
plt.hist(df_no_outliers['ccs'],
         bins=np.arange(start=min(df_no_outliers['ccs'])-0.5, stop=max(df_no_outliers['ccs'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xticks(ticks=np.arange(min(df_no_outliers['ccs']), max(df_no_outliers['ccs'])+1, 3))
plt.ylabel('n of people')
plt.title('Histogram of CCS score')

# plot bar charts for categorical variables
plt.figure()
sns.countplot(x='gender', data=df_categorical)
plt.title('Bar chart of Gender')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Male', 'Female', 'Non-Binary', 'Prefer not to say'])
plt.ylabel('Count')

plt.figure()
sns.countplot(x='education', data=df_categorical)
plt.title('Bar chart of education')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=['Elementary\nschool', 'Middle\nschool', 'High\nschool', 'Bachelor', 'Master', 'Doctoral'])
plt.ylabel('Count')

plt.figure()
sns.countplot(x='marital', data=df_categorical)
plt.title('Bar chart of marital')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=['Single', 'Married', 'Divorced', 'Widowed', 'Separated', 'Prefer not\nto say'])
plt.ylabel('Count')

# Bi-variate analysis:
# Stacked bar charts for pairs of categorical variables
# combinations of two variables in both directions
combinations = list(itertools.permutations(df_categorical.columns, 2))
# create stacked bar charts for each pair of variables using a for loop
for combination in combinations:
    fig, ax = plt.subplots(figsize=(8, 6))
    pivot_table = df.groupby(list(combination)).size().unstack()
    pivot_table.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f"Stacked Bar Chart for {combination[0]} vs {combination[1]}")
    ax.set_xlabel(combination[0])
    ax.set_ylabel('Count')
    ax.legend(title=combination[1])
    plt.tight_layout()

# Pair plot for numerical variables
plt.figure()
sns.pairplot(df_no_outliers)

# perform Kolmogorov-Smirnov test on numerical variables, to test normality
for col in df_numerical.columns:
    ks_statistic, ks_p_value = kstest(df_numerical[col], 'norm')
    print(f'Kolmogorov-Smirnov test for {col}:')
    print(f'KS Statistic: {ks_statistic}')
    print(f'p-value: {ks_p_value}')
# based on the results, we reject the hypothesis that data is normally distributed

# Multivariate analysis:
# study correlation between all the variables by plotting the heatmap of the whole dataframe
plt.figure()
sns.heatmap(pd.concat([df_categorical, df_no_outliers], axis=1).corr(), vmin=-1, vmax=1, center=0, cmap='Spectral', annot=True)

# we can see that the CCS variable is strongly correlated with the other questionnaires
# also, we will focus our game on mental health, so we decided to drop CCS
# we also saw that results were better without it
df_no_outliers.drop('ccs', axis=1, inplace=True)

# DATA PREPARATION
# One hot encoding of categorical columns, in order to perform PCA
encoder = OneHotEncoder(sparse_output=False)
categorical_columns = ['gender', 'education', 'marital']
categorical_encoded = encoder.fit_transform(df_sum[categorical_columns])
categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_columns))
print('\ncategorical_encoded_df:')
print(categorical_encoded_df)

# Replacing education values with [0, 1, 2, 3, 4, 5] to perform FAMD
education_mapping = {
    5: 0,
    8: 1,
    13: 2,
    18: 3,
    22: 4,
    25: 5
}
df_categorical.loc[:, 'education'] = df_categorical['education'].map(education_mapping)
print('\ndf_categorical after education mapping:')
print(df_categorical)


# create dataframe that we will use to perform PCA: cleaned numerical data + one hot encoded categorical data
df_tot_pca = pd.concat([categorical_encoded_df, df_no_outliers], axis=1)
# scale data using the robust scaler
df_all_pca = RobustScaler().fit_transform(df_tot_pca)
df_all_pca = pd.DataFrame(df_all_pca, columns=[col for col in df_tot_pca.columns])

# create dataframe that we will use to perform FAMD: cleaned numerical data + categorical data (with mapped education)
df_tot_famd = pd.concat([df_categorical, df_no_outliers], axis=1)
# scale data using the robust scaler
df_all_famd = RobustScaler().fit_transform(df_tot_famd)
df_all_famd = pd.DataFrame(df_all_famd, columns=[col for col in df_tot_famd.columns])

# PCA
# first, perform PCA to print the overall explained variance and find out the optimal number of components
pca = PCA()
pca.fit(df_all_pca)
variance = pca.explained_variance_ratio_.cumsum()
plt.figure(12)
print(f'Comulative explained variance PCA: {variance}')
plt.plot(range(1, len(variance) + 1), variance, marker='o')
plt.title('Cumulative Explained Variance PCA')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')

# then, we apply PCA with 7 components (that explain ~80% of the variance)
pca = PCA(n_components=7)
data_transformed_pca = pca.fit_transform(df_all_pca)
df_transformed_pca = pd.DataFrame(data_transformed_pca)

# FAMD
# Convert each column to 'category' type
categorical_columns = ["gender", "education", "marital"]
for column in categorical_columns:
    df_all_famd[categorical_columns] = df_all_famd[categorical_columns].astype('category')

# first, perform FAMD with all the features,
# to print the overall explained variance and find out the optimal number of components
famd = FAMD(n_components=9, n_iter=7, random_state=42)
famd.fit_transform(df_all_famd)
eigenvalues = famd.eigenvalues_
explained_variance = eigenvalues / eigenvalues.sum()
cumulative_variance=np.cumsum(explained_variance)
print(f' Cumulative Explained Variance FAMD: {cumulative_variance}')
plt.figure()
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title('Cumulative Explained Variance FAMD')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')

# then, we apply FAMD with 6 components (that explain ~78% of the variance)
famd = FAMD(n_components=6, n_iter=7, random_state=42)
data_transformed_famd = famd.fit_transform(df_all_famd)
df_transformed_famd = pd.DataFrame(data_transformed_famd)

# CLUSTERING
# In order to define the optimal number of clusters, we can use both the graphical and the analytical method:

# Elbow method for each one of the spaces we have:
# (1) No preprocessing, using the one hot encoded df
inertia = []
k_values = range(2, 11)
for k in k_values:
    kmedoids = KMedoids(n_clusters=k,
                        random_state=0,
                        init='k-medoids++',
                        metric='seuclidean',
                        method='pam',
                        max_iter=100)
    kmedoids.fit(df_all_pca)
    inertia.append(kmedoids.inertia_)
plt.figure()
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for no pre-processing, one hot encoded')
plt.grid()

# (2) No preprocessing, using the df with mapped education
inertia = []
k_values = range(2, 11)
for k in k_values:
    kmedoids = KMedoids(n_clusters=k,
                        random_state=0,
                        init='k-medoids++',
                        metric='seuclidean',
                        method='pam',
                        max_iter=100)
    kmedoids.fit(df_all_famd)
    inertia.append(kmedoids.inertia_)
plt.figure()
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for no pre-processing, mapped education')
plt.grid()

# (3) PCA
inertia = []
k_values = range(2, 11)
for k in k_values:
    kmedoids = KMedoids(n_clusters=k,
                        random_state=0,
                        init='k-medoids++',
                        metric='seuclidean',
                        method='pam',
                        max_iter=100)
    kmedoids.fit(df_transformed_pca)
    inertia.append(kmedoids.inertia_)
plt.figure()
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for PCA')
plt.grid()

# (4) FAMD
inertia = []
k_values = range(2, 11)
for k in k_values:
    kmedoids = KMedoids(n_clusters=k,
                        random_state=0,
                        init='k-medoids++',
                        metric='seuclidean',
                        method='pam',
                        max_iter=100)
    kmedoids.fit(df_transformed_famd)
    inertia.append(kmedoids.inertia_)
plt.figure()
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for FAMD')
plt.grid()

# Analytical method: evaluate silhouette scores
datasets = {  # store dataframes and names in a dictionary
    'None OHE': df_all_pca,
    'None EduMap': df_all_famd,
    'PCA': df_transformed_pca,
    'FAMD': df_transformed_famd
}
# silhouette scores for different numbers of clusters for each dataset
plt.figure()
for method, data in datasets.items():
    silhouette_scores = []
    for k in range(2, 11):
        kmedoids = KMedoids(n_clusters=k,
                            random_state=0,
                            init='k-medoids++',
                            metric='seuclidean',
                            method='pam',
                            max_iter=100)
        labels = kmedoids.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))
    plt.plot(range(2, 11), silhouette_scores, label=method, marker='o')
plt.title('Silhouette Score for Different Numbers of Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.legend()
plt.grid()

# Another analytical method to plot the silhouette scores:
# Plot silhouette analysis for each dataset
for method, data in datasets.items():
    range_n_clusters = [2, 3, 4, 5, 6]  # we take a smaller range than before
    for k in range_n_clusters:
        plt.figure()
        kmedoids = KMedoids(n_clusters=k,
                            random_state=0,
                            init='k-medoids++',
                            metric='seuclidean',
                            method='pam',
                            max_iter=100)
        labels = kmedoids.fit_predict(data)
        silhouette_avg = silhouette_score(data, labels)
        print(
            "For n_clusters =",
            k,
            "The average silhouette_score is :",
            silhouette_avg,
        )
        # compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(data, labels)
        y_lower = 10
        for i in range(k):
            # aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            plt.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        plt.title(f"Silhouette analysis on {method} with n_clusters = {k}")
        plt.xlabel("The silhouette coefficient values")
        plt.ylabel("Cluster label")
        # vertical line for average silhouette score of all the values
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# actually apply K-Medoids algorithm
# extensively justify choice here !!!!!!!!!!!!!!!!!!!!
kmedoids = KMedoids(n_clusters=3,
                    random_state=0,
                    init='k-medoids++',
                    metric='seuclidean',
                    method='pam',
                    max_iter=100)
labels = kmedoids.fit_predict(df_transformed_pca)

# add the cluster labels to the original numerical dataframe
df_numerical['Cluster'] = labels
print(df_numerical)
# add the cluster labels to the original categorical dataframe
df_categorical['Cluster'] = labels
print(df_categorical)

# STATISTICAL ANALYSIS
# Numerical attributes:
# we know from Kolmogorov-Smirnov test that data is not normally distributed => perform non-parametric tests
# Kruskal-Wallis Test
feature_numerical = [col for col in df_numerical.columns if col != "Cluster"]
for feature in feature_numerical:
    groups = [df_numerical[df_numerical['Cluster'] == cluster][[feature]] for cluster in df_numerical['Cluster'].unique()]
    stat, p = stats.kruskal(*groups)
    print(f"Kruskal-Wallis Test for {feature}:")
    print("Kruskal-Wallis H-statistic:", stat)
    print("p-value:", p)

# Pairwise Mann-Whitney U test
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
print('\nResults of Pairwise Mann-Whitney U test:')
print(results_df)

# Categorical attributes:
# create contingency tables
# the condition for the Pairwise Chi-Square Test is not respected (they have less than 5 counts for some cells)
# save contingency tables as csv files, to perform Fisher's test for contingency tables bigger that 2x2 in R
columns_categorical = [col for col in df_categorical.columns if col != "Cluster"]
for column in columns_categorical:
    contingency_table = pd.crosstab(df_categorical[column], df_categorical['Cluster'])
    print(f"\nContingency Table for {column}:")
    print(contingency_table)
    print(f"Shape: {contingency_table.shape}")
    contingency_table.to_csv(f"{column}_contingency_table.csv", index=True, header=True)
# run Fisher.R file to see results (p value is < 0.001)

# mean numerical values for each cluster
cluster_summary = df_numerical.groupby('Cluster').mean()
print(cluster_summary)

plt.show()