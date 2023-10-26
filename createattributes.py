import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

df = pd.read_csv('project_data/dataset_project_eHealth20232024.csv')
print(df.info)
print(f'nan in the df: {df.isnull().sum().sum()}')
print(f'rows with at least 1 nan: {df.isnull().T.any().T.sum()}')
print(f'columns with at least 1 nan: \n{df.isnull().any()}')

mean_age = df['age'].mean()
df['age'].fillna(value=mean_age, inplace=True)
print(df.mode().iloc[0])
df.fillna(value=df.mode().iloc[0], inplace=True)

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
         bins=np.arange(start=min(df_sum['income']), stop=max(df_sum['income'])+1, step=5000),
         color='skyblue', ec='blue')
plt.xlabel('income')
plt.xticks(ticks=np.arange(min(df_sum['income']), max(df_sum['income'])+1, 5000))
plt.ylabel('n of people')
plt.title('income')

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
# drop the gender as we see that it's not correlated to anything: this may be because we have a lot of "prefer not to say"
df_sum.drop('gender', axis=1, inplace=True)

# pca:
# scale data
df_sum = StandardScaler().fit_transform(df_sum)
# normalize ??
# df_sum = normalize(df_sum)
# perform pca to find out optimal number of components
pca = PCA()
pca.fit(df_sum)
variance = pca.explained_variance_ratio_.cumsum()
plt.figure(12)
print(variance)
plt.plot(range(1, len(variance) + 1), variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance Plot')

plt.show()
# try with n=4
pca =PCA(n_components=4)
pca.fit(df_sum)

# try with n=5
pca =PCA(n_components=5)
pca.fit(df_sum)