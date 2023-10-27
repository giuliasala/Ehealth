import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('project_data/dataset_project_eHealth20232024.csv')
print(df.info)
print(f'nan in the df: {df.isnull().sum().sum()}')
print(f'rows with at least 1 nan: {df.isnull().T.any().T.sum()}')
print(f'columns with at least 1 nan: \n{df.isnull().any()}')

print(df.mode().iloc[0])
df.fillna(value=df.mode().iloc[0], inplace=True)

df.drop_duplicates(keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.info)

plt.figure(1)
plt.hist(df['age'],
         bins=np.arange(start=min(df['age']), stop=max(df['age'])+1, step=5),  # distribute the bars every n(=step) years
         color='skyblue', ec='blue')
plt.xlabel('age')
plt.xticks(ticks=np.arange(min(df['age']), max(df['age'])+1, 5))  # adapt tick frequency on x axis
plt.ylabel('n of people')
plt.title('age')

plt.figure(2)
plt.hist(df['gender'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('gender')
labels_gen = ['male', 'female', 'non binary', 'prefer not to say']
plt.xticks(ticks=[0, 1, 2, 3], labels=labels_gen)
plt.ylabel('n of people')
plt.title('gender')

plt.figure(3)
bin_centers = [np.mean([bin_left, bin_right]) for bin_left, bin_right in zip([0, 5, 8, 13, 18, 22, 25], [5, 8, 13, 18, 22, 25, 28])]
plt.hist(df['education'], bins=bin_centers, color='skyblue', ec='blue')
plt.xlabel('education')
labels_edu = ['elem.', 'middle', 'high', 'bachelor', 'master', 'doctoral']
plt.xticks(ticks=[5, 8, 13, 18, 22, 25], labels=labels_edu)
plt.ylabel('n of people')
plt.title('education')

plt.figure(4)
plt.hist(df['marital'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color='skyblue', ec='blue')
plt.xlabel('marital status')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=['single', 'married', 'divorced', 'widowed', 'separated', 'pnts'])
plt.ylabel('n of people')
plt.title('marital status')

plt.figure(5)
plt.hist(df['income'],
         bins=np.arange(start=min(df['income']), stop=max(df['income'])+1, step=5000),
         color='skyblue', ec='blue')
plt.xlabel('income')
plt.xticks(ticks=np.arange(min(df['income']), max(df['income'])+1, 5000))
plt.ylabel('n of people')
plt.title('income')

plt.figure(6)
plt.hist(df['phq_1'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('phq_1')
labels_gen = ['never', 'same days', 'more than\nhalf', 'almost every day']
plt.xticks(ticks=[0, 1, 2, 3], labels=labels_gen)
plt.ylabel('n of people')
plt.title('phq_1')

plt.figure(7)
plt.hist(df['phq_2'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('phq_2')
labels_gen = ['never', 'same days', 'more than\nhalf', 'almost every day']
plt.xticks(ticks=[0, 1, 2, 3], labels=labels_gen)
plt.ylabel('n of people')
plt.title('phq_2')

plt.figure(8)
plt.hist(df['phq_3'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('phq_3')
labels_gen = ['never', 'same days', 'more than\nhalf', 'almost every day']
plt.xticks(ticks=[0, 1, 2, 3], labels=labels_gen)
plt.ylabel('n of people')
plt.title('phq_3')

plt.figure(9)
plt.hist(df['phq_4'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('phq_4')
labels_gen = ['never', 'same days', 'more than\nhalf', 'almost every day']
plt.xticks(ticks=[0, 1, 2, 3], labels=labels_gen)
plt.ylabel('n of people')
plt.title('phq_4')

plt.figure(10)
plt.hist(df['phq_5'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('phq_5')
labels_gen = ['never', 'same days', 'more than\nhalf', 'almost every day']
plt.xticks(ticks=[0, 1, 2, 3], labels=labels_gen)
plt.ylabel('n of people')
plt.title('phq_5')

plt.figure(11)
plt.hist(df['phq_6'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('phq_6')
labels_gen = ['never', 'same days', 'more than\nhalf', 'almost every day']
plt.xticks(ticks=[0, 1, 2, 3], labels=labels_gen)
plt.ylabel('n of people')
plt.title('phq_6')

plt.figure(12)
plt.hist(df['phq_7'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('phq_7')
labels_gen = ['never', 'same days', 'more than\nhalf', 'almost everyday']
plt.xticks(ticks=[0, 1, 2, 3], labels=labels_gen)
plt.ylabel('n of people')
plt.title('phq_7')

plt.figure(13)
plt.hist(df['phq_8'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('phq_8')
labels_gen = ['never', 'same days', 'more than\nhalf', 'almost everyday']
plt.xticks(ticks=[0, 1, 2, 3], labels=labels_gen)
plt.ylabel('n of people')
plt.title('phq_8')

plt.figure(14)
plt.hist(df['phq_9'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('phq_9')
labels_gen = ['never', 'same days', 'more than\nhalf', 'almost every day']
plt.xticks(ticks=[0, 1, 2, 3], labels=labels_gen)
plt.ylabel('n of people')
plt.title('phq_9')

plt.figure(15)
plt.hist(df['gad_1'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('gad_1')
plt.xticks(ticks=[0, 1, 2, 3], labels=['never', 'some days', 'more than\nhalf', 'almost everyday'])
plt.ylabel('n of people')
plt.title('gad_1')

plt.figure(16)
plt.hist(df['gad_2'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('gad_2')
plt.xticks(ticks=[0, 1, 2, 3], labels=['never', 'some days', 'more than\nhalf', 'almost everyday'])
plt.ylabel('n of people')
plt.title('gad_2')

plt.figure(17)
plt.hist(df['gad_3'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('gad_3')
plt.xticks(ticks=[0, 1, 2, 3], labels=['never', 'some days', 'more than\nhalf', 'almost everyday'])
plt.ylabel('n of people')
plt.title('gad_3')

plt.figure(18)
plt.hist(df['gad_4'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('gad_4')
plt.xticks(ticks=[0, 1, 2, 3], labels=['never', 'some days', 'more than\nhalf', 'almost everyday'])
plt.ylabel('n of people')
plt.title('gad_4')

plt.figure(19)
plt.hist(df['gad_5'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('gad_5')
plt.xticks(ticks=[0, 1, 2, 3], labels=['never', 'some days', 'more than\nhalf', 'almost everyday'])
plt.ylabel('n of people')
plt.title('gad_5')

plt.figure(20)
plt.hist(df['gad_6'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('gad_6')
plt.xticks(ticks=[0, 1, 2, 3], labels=['never', 'some days', 'more than\nhalf', 'almost everyday'])
plt.ylabel('n of people')
plt.title('gad_6')

plt.figure(21)
plt.hist(df['gad_7'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('gad_7')
plt.xticks(ticks=[0, 1, 2, 3], labels=['never', 'some days', 'more than\nhalf', 'almost everyday'])
plt.ylabel('n of people')
plt.title('gad_7')

plt.figure(22)
plt.hist(df['eheals_1'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color='skyblue', ec='blue')
plt.xlabel('eheals_1')
plt.xticks(ticks=[1, 2, 3, 4, 5], labels=['strongly\ndisagree', 'disagree', 'nether agree\nnor disagree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('eheals_1')

plt.figure(23)
plt.hist(df['eheals_2'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color='skyblue', ec='blue')
plt.xlabel('eheals_2')
plt.xticks(ticks=[1, 2, 3, 4, 5], labels=['strongly\ndisagree', 'disagree', 'nether agree\nnor disagree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('eheals_2')

plt.figure(24)
plt.hist(df['eheals_3'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color='skyblue', ec='blue')
plt.xlabel('eheals_3')
plt.xticks(ticks=[1, 2, 3, 4, 5], labels=['strongly\ndisagree', 'disagree', 'nether agree\nnor disagree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('eheals_3')

plt.figure(25)
plt.hist(df['eheals_4'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color='skyblue', ec='blue')
plt.xlabel('eheals_4')
plt.xticks(ticks=[1, 2, 3, 4, 5], labels=['strongly\ndisagree', 'disagree', 'nether agree\nnor disagree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('eheals_4')

plt.figure(26)
plt.hist(df['eheals_5'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color='skyblue', ec='blue')
plt.xlabel('eheals_5')
plt.xticks(ticks=[1, 2, 3, 4, 5], labels=['strongly\ndisagree', 'disagree', 'nether agree\nnor disagree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('eheals_5')

plt.figure(27)
plt.hist(df['eheals_6'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color='skyblue', ec='blue')
plt.xlabel('eheals_6')
plt.xticks(ticks=[1, 2, 3, 4, 5], labels=['strongly\ndisagree', 'disagree', 'nether agree\nnor disagree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('eheals_6')

plt.figure(28)
plt.hist(df['eheals_7'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color='skyblue', ec='blue')
plt.xlabel('eheals_7')
plt.xticks(ticks=[1, 2, 3, 4, 5], labels=['strongly\ndisagree', 'disagree', 'nether agree\nnor disagree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('eheals_7')

plt.figure(29)
plt.hist(df['eheals_8'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color='skyblue', ec='blue')
plt.xlabel('eheals_8')
plt.xticks(ticks=[1, 2, 3, 4, 5], labels=['strongly\ndisagree', 'disagree', 'nether agree\nnor disagree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('eheals_8')

plt.figure(30)
plt.hist(df['heas_1'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('heas_1')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Never', 'Some days', 'More than \nhalf', 'Almost every \nday'])
plt.ylabel('n of people')
plt.title('heas_1')

plt.figure(31)
plt.hist(df['heas_2'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('heas_2')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Never', 'Some days', 'More than \nhalf', 'Almost every \nday'])
plt.ylabel('n of people')
plt.title('heas_2')

plt.figure(32)
plt.hist(df['heas_3'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('heas_3')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Never', 'Some days', 'More than \nhalf', 'Almost every \nday'])
plt.ylabel('n of people')
plt.title('heas_3')

plt.figure(33)
plt.hist(df['heas_4'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('heas_4')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Never', 'Some days', 'More than \nhalf', 'Almost every \nday'])
plt.ylabel('n of people')
plt.title('heas_4')

plt.figure(34)
plt.hist(df['heas_5'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('heas_5')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Never', 'Some days', 'More than \nhalf', 'Almost every \nday'])
plt.ylabel('n of people')
plt.title('heas_5')

plt.figure(35)
plt.hist(df['heas_6'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('heas_6')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Never', 'Some days', 'More than \nhalf', 'Almost every \nday'])
plt.ylabel('n of people')
plt.title('heas_6')

plt.figure(36)
plt.hist(df['heas_7'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('heas_7')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Never', 'Some days', 'More than \nhalf', 'Almost every \nday'])
plt.ylabel('n of people')
plt.title('heas_7')

plt.figure(37)
plt.hist(df['heas_8'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('heas_8')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Never', 'Some days', 'More than \nhalf', 'Almost every \nday'])
plt.ylabel('n of people')
plt.title('heas_8')

plt.figure(38)
plt.hist(df['heas_9'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('heas_9')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Never', 'Some days', 'More than \nhalf', 'Almost every \nday'])
plt.ylabel('n of people')
plt.title('heas_9')

plt.figure(39)
plt.hist(df['heas_10'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('heas_10')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Never', 'Some days', 'More than \nhalf', 'Almost every \nday'])
plt.ylabel('n of people')
plt.title('heas_10')

plt.figure(40)
plt.hist(df['heas_11'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('heas_11')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Never', 'Some days', 'More than \nhalf', 'Almost every \nday'])
plt.ylabel('n of people')
plt.title('heas_11')

plt.figure(41)
plt.hist(df['heas_12'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('heas_12')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Never', 'Some days', 'More than \nhalf', 'Almost every \nday'])
plt.ylabel('n of people')
plt.title('heas_12')

plt.figure(42)
plt.hist(df['heas_13'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='skyblue', ec='blue')
plt.xlabel('heas_13')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Never', 'Some days', 'More than \nhalf', 'Almost every \nday'])
plt.ylabel('n of people')
plt.title('heas_13')

plt.figure(43)
plt.hist(df['ccs_1'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], color='skyblue', ec='blue')
plt.xlabel('ccs_1')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['strongly\ndisagree', 'disagree', 'somewhat\ndisagree', 'neither agree\nnor disagree', 'somewhat agree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('ccs_1')

plt.figure(44)
plt.hist(df['ccs_2'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], color='skyblue', ec='blue')
plt.xlabel('ccs_2')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['strongly\ndisagree', 'disagree', 'somewhat\ndisagree', 'neither agree\nnor disagree', 'somewhat agree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('ccs_2')

plt.figure(45)
plt.hist(df['ccs_3'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], color='skyblue', ec='blue')
plt.xlabel('ccs_3')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['strongly\ndisagree', 'disagree', 'somewhat\ndisagree', 'neither agree\nnor disagree', 'somewhat agree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('ccs_3')

plt.figure(46)
plt.hist(df['ccs_4'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], color='skyblue', ec='blue')
plt.xlabel('ccs_4')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['strongly\ndisagree', 'disagree', 'somewhat\ndisagree', 'neither agree\nnor disagree', 'somewhat agree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('ccs_4')

plt.figure(47)
plt.hist(df['ccs_5'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], color='skyblue', ec='blue')
plt.xlabel('ccs_5')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['strongly\ndisagree', 'disagree', 'somewhat\ndisagree', 'neither agree\nnor disagree', 'somewhat agree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('ccs_5')

plt.figure(48)
plt.hist(df['ccs_6'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], color='skyblue', ec='blue')
plt.xlabel('ccs_6')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['strongly\ndisagree', 'disagree', 'somewhat\ndisagree', 'neither agree\nnor disagree', 'somewhat agree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('ccs_6')

plt.figure(49)
plt.hist(df['ccs_7'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], color='skyblue', ec='blue')
plt.xlabel('ccs_7')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['strongly\ndisagree', 'disagree', 'somewhat\ndisagree', 'neither agree\nnor disagree', 'somewhat agree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('ccs_7')

plt.figure(50)
plt.hist(df['ccs_8'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], color='skyblue', ec='blue')
plt.xlabel('ccs_8')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['strongly\ndisagree', 'disagree', 'somewhat\ndisagree', 'neither agree\nnor disagree', 'somewhat agree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('ccs_8')

plt.figure(51)
plt.hist(df['ccs_9'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], color='skyblue', ec='blue')
plt.xlabel('ccs_9')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['strongly\ndisagree', 'disagree', 'somewhat\ndisagree', 'neither agree\nnor disagree', 'somewhat agree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('ccs_9')

plt.figure(52)
plt.hist(df['ccs_10'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], color='skyblue', ec='blue')
plt.xlabel('ccs_10')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['strongly\ndisagree', 'disagree', 'somewhat\ndisagree', 'neither agree\nnor disagree', 'somewhat agree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('ccs_10')

plt.figure(53)
plt.hist(df['ccs_11'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], color='skyblue', ec='blue')
plt.xlabel('ccs_11')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['strongly\ndisagree', 'disagree', 'somewhat\ndisagree', 'neither agree\nnor disagree', 'somewhat agree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('ccs_11')

plt.figure(54)
plt.hist(df['ccs_12'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], color='skyblue', ec='blue')
plt.xlabel('ccs_12')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['strongly\ndisagree', 'disagree', 'somewhat\ndisagree', 'neither agree\nnor disagree', 'somewhat agree', 'agree', 'strongly agree'])
plt.ylabel('n of people')
plt.title('ccs_12')

# study correlation
plt.figure(55)
sns.heatmap(df.corr(), vmin=-1, vmax=1, center=0, cmap='Spectral')
plt.show()

# scale data ?
df_sum = StandardScaler().fit_transform(df_sum)

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
pca3 = PCA(n_components=3)
df3 = pca3.fit_transform(df_sum)
df_3 = pd.DataFrame(df3, columns=['pc1', 'pc2', 'pc3'])
print(df_3.info)
# try with n=4
pca4 = PCA(n_components=4)
df4 = pca4.fit_transform(df_sum)
df_4 = pd.DataFrame(df4, columns=['pc1', 'pc2', 'pc3', 'pc4'])
print(df_4.info)
# try with n=5
pca5 = PCA(n_components=5)
df5 = pca5.fit_transform(df_sum)
df_5 = pd.DataFrame(df5, columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5'])
print(df_5.info)

# clustering k-medoids