# import cvs files with the contingency tables of the categorical variables that we created in pycharm
gender <- read.csv("gender_contingency_table.csv", header = FALSE, skip = 1, row.names = 1)
education <- read.csv("education_contingency_table.csv", header = FALSE, skip = 1, row.names = 1)
marital <- read.csv("marital_contingency_table.csv", header = FALSE, skip = 1, row.names = 1)

# perform Fisher's test:
# increase the workspace, in order to be able to perform the exact test
# alternatively, if it takes too long, use simulate.p.value=TRUE and B=number of simulations
fisher_result_gender <- fisher.test(gender, workspace=90000000)
fisher_result_education <- fisher.test(education, workspace=90000000)
fisher_result_marital <- fisher.test(marital, workspace=90000000)

# print results
print(fisher_result_gender)
print(fisher_result_education)
print(fisher_result_marital)

# extract the pvalue
p_value_gender <- fisher_result_gender$p.value
p_value_education <- fisher_result_education$p.value
p_value_marital <- fisher_result_marital$p.value

# apply the Bonferroni correction
p_values <- c(p_value_gender, p_value_education, p_value_marital)
num_tests <- length(p_values)
alpha <- 0.05
bonferroni_corrected <- p_values * num_tests
significant_results <- bonferroni_corrected < alpha
print(bonferroni_corrected)
print(significant_results)
