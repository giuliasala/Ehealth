# import all the libraries that we need
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from prince import FAMD
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn_extra.cluster import KMedoids
from scipy import stats
from statsmodels.stats.multitest import multipletests
from scipy.stats import kstest, chi2_contingency
from matplotlib.backends.backend_pdf import PdfPages

# As we will print more than 20 (default value) figures at once,
# we changed the setting of the warning for too many figures open
plt.rcParams['figure.max_open_warning'] = 30

# Create an empty list where we are going to store our figures (using the .append everytime),
# in order to save them in a pdf file at the end
all_figures = []

# Read the cvs file
df = pd.read_csv('../project_data/dataset_project_eHealth20232024.csv')
# Print the dataframe
print('\ndataframe:')
print(df)


# DATA CLEANING
# Print the number NaN in the dataframe, the number of rows and the columns that contain NaN values
print(f'\nnan in the df: {df.isnull().sum().sum()}')
print(f'\nrows with at least 1 nan: {df.isnull().T.any().T.sum()}')
print(f'\ncolumns with at least 1 nan: \n{df.isnull().sum()}')
# Replace the NaN with the median value of each column
print('\nmedian values:')
print(df.median())
df.fillna(value=df.median(), inplace=True)

# Drop the duplicate rows in the dataframe
df.drop_duplicates(keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
print('\ndataframe after dropping duplicates:')
print(df)

# Create one attribute for each questionnaire, by summing all the scores of the questions:
# PHQ
col_phq = ['phq_1', 'phq_2', 'phq_3', 'phq_4', 'phq_5', 'phq_6', 'phq_7', 'phq_8', 'phq_9']
sum_phq = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_phq].sum()
    sum_phq.append(row_sum)
print(f'\nPHQ sum: {sum_phq}')

# GAD
col_gad = ['gad_1', 'gad_2', 'gad_3', 'gad_4', 'gad_5', 'gad_6', 'gad_7']
sum_gad = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_gad].sum()
    sum_gad.append(row_sum)
print(f'\nGAD sum: {sum_gad}')

# eHEALS
col_eheals = ['eheals_1', 'eheals_2', 'eheals_3', 'eheals_4', 'eheals_5', 'eheals_6', 'eheals_7', 'eheals_8']
sum_eheals = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_eheals].sum()
    sum_eheals.append(row_sum)
print(f'\nEHEALS sum: {sum_eheals}')

# HEAS
col_heas = ['heas_1', 'heas_2', 'heas_3', 'heas_4', 'heas_5', 'heas_6', 'heas_7', 'heas_8', 'heas_9', 'heas_10',
            'heas_11', 'heas_12', 'heas_13']
sum_heas = []
for row_index in range(150):
    row_sum = df.loc[row_index, col_heas].sum()
    sum_heas.append(row_sum)
print(f'\nHEAS sum: {sum_heas}')

# CCS
# The higher the score, the more skeptic the person is about climate change
# except for the following questions
# For this reason, we reverse the scores of ccs 3, 6, 7 and 12 before summing
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
print(f'\nCCS sum: {sum_ccs}')

# Create the new dataframe (df_sum):
# we take the original columns for age, gender, education, marital and income (df1)
# and the new columns for the questionnaires (df2)
# then, we concatenate them
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
# The following option is set in order to see, from now on, all columns when printing the dataframes
pd.set_option('display.max_columns', None)
# Print the information about the new dataframe
print('\ndf_sum info:')
print(df_sum.info())
print(df_sum)

# Create two different dataframes for categorical variables (gender, education, marital)
# and numerical variables (age, income and the 5 questionnaires: phq, gad, eheals, heas, ccs)
df_numerical = df_sum.drop(columns=['gender', 'education', 'marital'])  # dropping the categorical
df_categorical = df_sum[['gender', 'education', 'marital']].copy()  # only copying the categorical
# Print the 2 sub-dataframes
print('\ndf_numerical:')
print(df_numerical)
print('\ndf_categorical:')
print(df_categorical)

# Identification of outliers:
# Boxplot of the numerical variables
attributes_to_plot = [col for col in df_numerical.columns]
fig1 = plt.figure(figsize=(10, 6 * len(attributes_to_plot)))
for i, attribute in enumerate(attributes_to_plot, 1):
    plt.subplot(len(attributes_to_plot), 1, i)
    plt.boxplot(df_numerical[attribute])
    plt.title(f'Boxplot of {attribute}')
    plt.ylabel('Values')
    plt.xlabel(attribute)
    plt.tight_layout()
all_figures.append(fig1)

# We can see that only income has outliers, now we have to handle them
# First, we identify outliers using the Inter-Quartile Range method
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
df_no_outliers['income'] = df_no_outliers['income'].apply(lambda x: upper_bound if x > upper_bound else x)
# Print the rows that had outliers to see the change
print('\nSame rows, values replaced:')
print(df_no_outliers.iloc[[28, 52]])


# EXPLORATORY DATA ANALYSIS
# Uni-variate analysis:
# See also boxplots plotted before for numerical variables

# Print summary statistics for numerical variables
print('\nSummary statistics for numerical variables:')
print(df_no_outliers.describe())

# Plot histograms for numerical variables:
# Age
fig2 = plt.figure()
plt.hist(df_no_outliers['age'],
         bins=np.arange(start=min(df_sum['age']), stop=max(df_sum['age'])+1, step=5),  # distribute the bars every n(=step) years
         color='skyblue', ec='blue')
plt.xticks(ticks=np.arange(min(df_no_outliers['age']), max(df_no_outliers['age'])+1, 5))  # adapt tick frequency on x axis
plt.ylabel('n of people')
plt.title('Histogram of age')
all_figures.append(fig2)

# Income
fig3 = plt.figure()
plt.hist(df_no_outliers['income'],
         bins=np.arange(start=min(df_no_outliers['income']), stop=max(df_no_outliers['income'])+5000, step=5000),
         color='skyblue', ec='blue')
plt.xticks(ticks=np.arange(min(df_no_outliers['income']), max(df_no_outliers['income'])+5000, 5000))
plt.ylabel('n of people')
plt.title('Histogram of income')
all_figures.append(fig3)

# PHQ
fig4 = plt.figure()
plt.hist(df_no_outliers['phq'],
         bins=np.arange(start=min(df_no_outliers['phq'])-0.5, stop=max(df_no_outliers['phq'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xticks(ticks=np.arange(min(df_no_outliers['phq']), max(df_no_outliers['phq'])+1, 1))
plt.ylabel('n of people')
plt.title('Histogram of PHQ score')
all_figures.append(fig4)

# GAD
fig5 = plt.figure()
plt.hist(df_no_outliers['gad'],
         bins=np.arange(start=min(df_no_outliers['gad'])-0.5, stop=max(df_no_outliers['gad'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xticks(ticks=np.arange(min(df_no_outliers['gad']), max(df_no_outliers['gad'])+1, 1))
plt.ylabel('n of people')
plt.title('Histogram of GAD score')
all_figures.append(fig5)

# eHEALS
fig6 = plt.figure()
plt.hist(df_no_outliers['eheals'],
         bins=np.arange(start=min(df_no_outliers['eheals'])-0.5, stop=max(df_no_outliers['eheals'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xticks(ticks=np.arange(min(df_no_outliers['eheals']), max(df_no_outliers['eheals'])+1, 2))
plt.ylabel('n of people')
plt.title('Histogram of EHEALS score')
all_figures.append(fig6)

# HEAS
fig7 = plt.figure()
plt.hist(df_no_outliers['heas'],
         bins=np.arange(start=min(df_no_outliers['heas'])-0.5, stop=max(df_no_outliers['heas'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xticks(ticks=np.arange(min(df_no_outliers['heas']), max(df_no_outliers['heas'])+1, 2))
plt.ylabel('n of people')
plt.title('Histogram of HEAS score')
all_figures.append(fig7)

# CCS
fig8 = plt.figure()
plt.hist(df_no_outliers['ccs'],
         bins=np.arange(start=min(df_no_outliers['ccs'])-0.5, stop=max(df_no_outliers['ccs'])+1.5, step=1),
         color='skyblue', ec='blue')
plt.xticks(ticks=np.arange(min(df_no_outliers['ccs']), max(df_no_outliers['ccs'])+1, 3))
plt.ylabel('n of people')
plt.title('Histogram of CCS score')
all_figures.append(fig8)

# Plot bar charts for categorical variables:
# Gender
fig9 = plt.figure()
sns.countplot(x='gender', data=df_categorical)
plt.title('Bar chart of Gender')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Male', 'Female', 'Non-Binary', 'Prefer not to say'])
plt.ylabel('Count')
all_figures.append(fig9)

# Education
fig10 = plt.figure()
sns.countplot(x='education', data=df_categorical)
plt.title('Bar chart of education')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=['Elementary\nschool', 'Middle\nschool', 'High\nschool', 'Bachelor', 'Master', 'Doctoral'])
plt.ylabel('Count')
all_figures.append(fig10)

# Marital status
fig11 = plt.figure()
sns.countplot(x='marital', data=df_categorical)
plt.title('Bar chart of marital')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=['Single', 'Married', 'Divorced', 'Widowed', 'Separated', 'Prefer not\nto say'])
plt.ylabel('Count')
all_figures.append(fig11)

# Bi-variate analysis:
# Stacked bar charts for pairs of categorical variables
# Create the stacked bar charts for each pair of variables using a for loop;
# to get the combinations of two variables in both directions, we use the permutation function
combinations = list(itertools.permutations(df_categorical.columns, 2))
for combination in combinations:
    fig, ax = plt.subplots()
    pivot_table = df.groupby(list(combination)).size().unstack()
    pivot_table.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f"Stacked Bar Chart for {combination[0]} vs {combination[1]}")
    ax.set_xlabel(combination[0])
    ax.set_ylabel('Count')
    ax.legend(title=combination[1])
    plt.tight_layout()
    all_figures.append(fig)

# Pair plot for numerical variables
pairplot = sns.pairplot(df_no_outliers)
fig12 = pairplot.fig
all_figures.append(fig12)

# Statistical test:
# Perform Kolmogorov-Smirnov test on numerical variables, to test normality
for col in df_numerical.columns:
    ks_statistic, ks_p_value = kstest(df_numerical[col], 'norm')
    print(f'\nKolmogorov-Smirnov test for {col}:')
    print(f'KS Statistic: {ks_statistic}')
    print(f'p-value: {ks_p_value}')
# Based on the results, we reject the hypothesis that data is normally distributed

# Multivariate analysis:
# Study correlation between all the variables by plotting the heatmap of the whole dataframe
fig13 = plt.figure()
heatmap = sns.heatmap(pd.concat([df_categorical, df_no_outliers], axis=1).corr(), vmin=-1, vmax=1, center=0,
                      cmap='Spectral', annot=True)
# rotate x-axis ticks by 30 degrees
for tick in heatmap.get_xticklabels():
    tick.set_rotation(30)
all_figures.append(fig13)

# From the heatmap, we can see that the CCS variable is strongly correlated with the other questionnaires;
# furthermore, we saw that the final results (silhouette scores and statistical tests) were better without it
# Also, we will focus our game on mental health, so we don't have that tipe of constraint.
# So, we decided to drop the CCS variable
df_no_outliers.drop('ccs', axis=1, inplace=True)


# DATA PREPARATION
# Our goal for this part is to perform both PCA and FAMD, on different dataframes, and then compare them.
# First, we need to work on the dataframes to make them suitable for performing the 2 data reduction techniques

# Perform one hot encoding for categorical columns: we will make use of this when we perform PCA
encoder = OneHotEncoder(sparse_output=False)
categorical_columns = ['gender', 'education', 'marital']
categorical_encoded = encoder.fit_transform(df_sum[categorical_columns])
categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_columns))
print('\ncategorical_encoded_df:')
print(categorical_encoded_df)

# Replacing education values with [0, 1, 2, 3, 4, 5]: we will make use of this when we perform FAMD
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

# Generate the 2 dataframes:
# Create dataframe that we will use to perform PCA:
# cleaned numerical data + one hot encoded categorical data
df_tot_pca = pd.concat([categorical_encoded_df, df_no_outliers], axis=1)
# scale data using the robust scaler
df_all_pca = RobustScaler().fit_transform(df_tot_pca)
df_all_pca = pd.DataFrame(df_all_pca, columns=[col for col in df_tot_pca.columns])

# Create dataframe that we will use to perform FAMD:
# cleaned numerical data + categorical data (with mapped education)
df_tot_famd = pd.concat([df_categorical, df_no_outliers], axis=1)
# scale data using the robust scaler
df_all_famd = RobustScaler().fit_transform(df_tot_famd)
df_all_famd = pd.DataFrame(df_all_famd, columns=[col for col in df_tot_famd.columns])

# PCA
# First, perform PCA without specifying the number of components;
# we did this in order to know the cumulative explained variance and to find out the optimal number of components
pca = PCA()
pca.fit(df_all_pca)
variance = pca.explained_variance_ratio_.cumsum()
fig14 = plt.figure()
print(f'\nComulative explained variance PCA: {variance}')
plt.plot(range(1, len(variance) + 1), variance, marker='o')
plt.title('Cumulative Explained Variance PCA')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
all_figures.append(fig14)

# Then, we apply PCA
# We decided to use 7 components, that explain ~80% of the variance
pca = PCA(n_components=7)
data_transformed_pca = pca.fit_transform(df_all_pca)
df_transformed_pca = pd.DataFrame(data_transformed_pca)

# FAMD
# Convert categorical columns to 'category' type
categorical_columns = ["gender", "education", "marital"]
for column in categorical_columns:
    df_all_famd[categorical_columns] = df_all_famd[categorical_columns].astype('category')

# First, perform FAMD with numeber of components = all the features we have;
# again, we did this in order to print the overall explained variance and to find out the optimal number of components
famd = FAMD(n_components=9, n_iter=7, random_state=42)
famd.fit_transform(df_all_famd)
eigenvalues = famd.eigenvalues_
explained_variance = eigenvalues / eigenvalues.sum()
cumulative_variance=np.cumsum(explained_variance)
print(f'\nCumulative Explained Variance FAMD: {cumulative_variance}')
fig15 = plt.figure()
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title('Cumulative Explained Variance FAMD')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
all_figures.append(fig15)

# Then, we apply FAMD
# We decided to use 6 components, that explain ~78% of the variance
famd = FAMD(n_components=6, n_iter=7, random_state=42)
data_transformed_famd = famd.fit_transform(df_all_famd)
df_transformed_famd = pd.DataFrame(data_transformed_famd)


# CLUSTERING
# In order to define the optimal number of clusters, we can use both the graphical and the analytical method:

# Elbow method for each one of the spaces we have, so we can compare them:
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
fig16 = plt.figure()
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for no pre-processing, one hot encoded')
plt.grid()
all_figures.append(fig16)

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
fig17 = plt.figure()
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for no pre-processing, mapped education')
plt.grid()
all_figures.append(fig17)

# (3) using the df to which we applied PCA
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
fig18 = plt.figure()
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for PCA')
plt.grid()
all_figures.append(fig18)

# (4) using the df to which we applied FAMD
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
fig19 = plt.figure()
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for FAMD')
plt.grid()
all_figures.append(fig19)

# Analytical method: evaluate silhouette scores
dataframes = {  # store dataframes and names in a dictionary
    # we use this when iterating through the dataframes to plot the silhouette for different number of clusters
    'None OHE': df_all_pca,
    'None EduMap': df_all_famd,
    'PCA': df_transformed_pca,
    'FAMD': df_transformed_famd
}

# Silhouette scores for different numbers of clusters, for each dataset
fig20 = plt.figure()
for method, data in dataframes.items():
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
all_figures.append(fig20)

# Another analytical method to plot the silhouette scores:
# Plot silhouette analysis for each dataset
for idx, (method, data) in enumerate(dataframes.items()):
    print(f'\nFor {method}:')
    fig, axs = plt.subplots(1, 5, figsize=(18, 5))
    fig.suptitle(f"Silhouette analysis on {method}", fontsize=16)
    range_n_clusters = [2, 3, 4, 5, 6]  # we take a smaller range than before
    for j, k in enumerate(range_n_clusters):
        ax = axs[j]
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
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                alpha=0.7,
            )
            # label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            # compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        ax.set_title(f"with n_clusters = {k}")
        ax.set_xlabel("Silhouette coefficient values")
        ax.set_ylabel("Cluster label")
        ax.set_yticks([])
        # vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    all_figures.append(fig)

# We applied the actual kmedoids algorithm with all possible combinations
# that we thought were promising (3 and 4 clusters, to the 4 dataframes)
# What we found was:
# For 4 clusters:
# all the dfs have more non-significant results in the statistical tests, meaning that 4 clusters are too much
# For 3 clusters:
# Dataframe with no preprocessing and mapped education (df_all_famd) -> gender has non-significant results,
# age, income and PHQ have 1 non-significant result over 3;
# Dataframe with no preprocessing and one hot encoding (df_all_pca) -> income has one non-significant result over 3,
# silhouette score for half of the samples of cluster 2 is < 0;
# Dataframe with PCA (df_transformed_pca) -> age and PHQ have one non-significant result over 3;
# Dataframe with FAMD (df_transformed_famd) -> age, GAD and HEAS have one non-significant result over 3.
# Considering these results we got
# and the EDA (ex: it makes sense that PHQ has one non-significant result, as we only have values until 9),
# we finally chose to apply the KMedoids with 3 clusters, on the df to which we applied PCA
kmedoids = KMedoids(n_clusters=3,
                    random_state=0,
                    init='k-medoids++',
                    metric='seuclidean',
                    method='pam',
                    max_iter=100)
labels = kmedoids.fit_predict(df_transformed_pca)

# Print number of samples in each cluster
print(f'\nNumber of samples in each cluster:{np.bincount(labels)}')

# Add the cluster labels to the dataframes:
# to the numerical dataframe (the one resulting from data processing, so without outliers and CCS)
df_no_outliers['Cluster'] = labels
print('\nNumerical df with cluster labels:')
print(df_no_outliers)
# to the categorical dataframe
df_categorical['Cluster'] = labels
print('\nCategorical df with cluster labels:')
print(df_categorical)


# STATISTICAL ANALYSIS
# Numerical attributes:
# we know from Kolmogorov-Smirnov test that data is not normally distributed => perform non-parametric tests

# Kruskal-Wallis Test
# Iterate through numerical variables and print their p-value
feature_numerical = [col for col in df_no_outliers.columns if col != "Cluster"]
for feature in feature_numerical:
    groups = [df_no_outliers[df_no_outliers['Cluster'] == cluster][[feature]] for cluster in df_no_outliers['Cluster'].unique()]
    stat, p = stats.kruskal(*groups)
    print(f"\nKruskal-Wallis Test for {feature}:")
    print("Kruskal-Wallis H-statistic:", stat)
    print("p-value:", p)

# Pairwise Mann-Whitney U test
# Iterate through numerical variables and each possible cluster combination. Then print the Bonferroni corrected p-values
feature_names = []
cluster1_values = []
cluster2_values = []
original_p_values = []
corrected_p_values = []
reject_hypothesis = []
for feature in feature_numerical:
    clusters = df_no_outliers['Cluster'].unique()
    for i, cluster1 in enumerate(clusters):
        for j, cluster2 in enumerate(clusters):
            if i >= j:
                # skip comparisons of a cluster with itself and duplicate comparisons
                continue
            group1 = df_no_outliers[df_no_outliers['Cluster'] == cluster1][feature]
            group2 = df_no_outliers[df_no_outliers['Cluster'] == cluster2][feature]
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
# Create a DataFrame to display all the results
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

# Create dictionary to store contingency tables
contingency_tables = {}
columns_categorical = [col for col in df_categorical.columns if col != "Cluster"]
for column in columns_categorical:
    contingency_table = pd.crosstab(df_categorical[column], df_categorical['Cluster'])
    print(f"\nContingency Table for {column}:")
    print(contingency_table)
    print(f"Shape: {contingency_table.shape}")
    # The condition for the Pairwise Chi-Square Test is not respected for education and marital
    # (they have less than 5 counts for some cells), but it is borderline for gender (only one cell with 4 counts)

    # Save contingency tables to the dictionary, so that we can then use the gender table outside the loop
    contingency_tables[column] = contingency_table
    # Save contingency tables as csv files, to perform Fisher's test in R (for contingency tables larger that 2x2)
    contingency_table.to_csv(f"{column}_contingency_table.csv", index=True, header=True)

# run Fisher.R file to see the results for all 3 variables (p value is < 0.001)

# Chi Square test for gender
overall_chi2_stat, overall_chi2_p_value, _, _ = chi2_contingency(contingency_tables['gender'])
print(f"\nChi-square Test for gender: Statistic - {overall_chi2_stat}, P-Value - {overall_chi2_p_value}")


# PERSONAS CREATION
# For each cluster:
# Extract median values, 25th and 75th percentile for numerical variables
cluster_summary = df_no_outliers.groupby('Cluster').agg(['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
print(f'\nNumerical summary:\n {cluster_summary}')
# Extract mode and percentages for categorical variables
# we can get them from the contingency tables
# See personas table

# Save in one pdf file all the figures that we have stored in the list all_figures
with PdfPages('EDA_figures.pdf') as pdf:
    for fig in all_figures:
        pdf.savefig(fig)
# Show all figures
plt.show()
