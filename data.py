import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Set display options for pandas
pd.set_option('display.max_columns', None)
# set option so more columns show on one row in the output
pd.set_option('display.width', 200)

# Load the dataset
data = pd.read_csv('pss-ascs-eng-2022-23-CSV.csv', low_memory=False)

# Filter data based on certain conditions and exclude non-respondents or certain stratum codes
data = data[
    (data['Response'] == 1) & (data['SupportSetting'] != 99) & (data['Stratum'] != 99) & (data['Stratum'].notnull())]

# Define columns to be dropped that are not necessary for the analysis
columns_to_drop = [
    'Geography_Code', 'Method of collection', 'Response',
    'An advocate was used to help to complete the questionnaire',
    'An interpreter was used to help to complete the questionnaire',
    'translated', 'Version of the questionnaire used', 'Service user is a replacement in the sample',
    'Responded to the original or reminder', 'Question 1 standard', 'Question 2 standard', 'Question 1 easy-read',
    'Question 2 easy-read',
    'Question 22',
    'Question 23a', 'Question 23b', 'Question 23c', 'Question 23d', 'Question 23e', 'Question 23f'
]

# Remove the columns defined above
data.drop(columns=columns_to_drop, inplace=True)

# Drop optional questions
optional_questions = ['Question 3b', 'Question 2b', 'Question 2c', 'Question 4b', 'Question 5b', 'Question 6b',
                      'Question 8b', 'Question 9b']
data.drop(columns=optional_questions, inplace=True)


def extended_describe(dataframe):
    """ Generate extended descriptive statistics including median and mode for numeric columns. """
    numeric_df = dataframe.select_dtypes(include=[np.number])
    description = dataframe.describe()
    medians = numeric_df.median().rename('median').to_frame().T
    modes = numeric_df.mode().iloc[0].rename('mode').to_frame().T
    extended_description = pd.concat([description, medians, modes])
    return extended_description


print("Analysis of filtered data:")
print(extended_describe(data))
print(data.head())

# Remove rows where 'ethgrp' is NaN
data = data.dropna(subset=['ethgrp'])


def impute_with_group_median_mode(data, group_column, method='mode'):
    """ Impute missing data using median or mode stratified by the specified group column. """
    for column in data.columns:
        if data[column].dtype == 'object' or column == group_column:
            continue
        if method == 'mode':
            mode_impute = lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            data[column] = data[column].fillna(data.groupby(group_column)[column].transform(mode_impute))
        else:
            data[column] = data[column].fillna(data.groupby(group_column)[column].transform('median'))
    return data


# Call function to impute data
impute_with_group_median_mode(data, 'Stratum', method='mode')

column_rename_map = {
    'LaCode': 'CASSR_code',
    'Gender': 'Gender',
    'ethgrp': 'Ethnicity',
    'agegrp': 'Age_group',
    'SupportSetting': 'Support_Setting',
    'Stratum': 'Stratum',
    'PSR': 'Primary_Support_Reason',
    'Question 1 combined': 'Q1_satisfaction_combined',
    'Question 2 combined': 'Q2_quality_of_life_combined',
    'Question 3a': 'Q3a_control_over_daily_life',
    'Question 4a': 'Q4a_clean_and_presentable',
    'Question 5a': 'Q5a_food_and_drink',
    'Question 6a': 'Q6a_clean_and_comfortable_home',
    'Question 7a': 'Q7a_feeling_safe',
    'Question 7c': 'Q7c_services_help_feeling_safe',
    'Question 8a': 'Q8a_social_contact',
    'Question 9a': 'Q9a_spend_time',
    'Question 10': 'Q10_help_impact_self_view',
    'Question 11': 'Q11_help_treatment_impact_self_view',
    'Question 12': 'Q12_feeling_lonely',
    'Question 13': 'Q13_finding_information_and_advice',
    'Question 14': 'Q14_general_health',
    'Question 15a': 'Q15a_pain_or_discomfort',
    'Question 15b': 'Q15b_anxiety_or_depression',
    'Question 16a': 'Q16a_manage_getting_around_indoors',
    'Question 16b': 'Q16b_manage_getting_in_out_bed_chair',
    'Question 16c': 'Q16c_manage_feeding_yourself',
    'Question 16d': 'Q16d_manage_finances_and_paperwork',
    'Question 17a': 'Q17a_manage_washing_yourself',
    'Question 17b': 'Q17b_manage_dressing_yourself',
    'Question 17c': 'Q17c_manage_using_toilet',
    'Question 17d': 'Q17d_manage_washing_face_and_hands',
    'Question 18': 'Q18_home_meeting_needs',
    'Question 19': 'Q19_getting_around_outside',
    'Question 20a': 'Q20a_practical_help_from_household',
    'Question 20b': 'Q20b_practical_help_from_other_household',
    'Question 20c': 'Q20c_no_practical_help',
    'Question 21a': 'Q21a_buy_additional_support_self',
    'Question 21b': 'Q21b_family_pays_additional_support',
    'Question 21c': 'Q21c_no_additional_support'
}

# Rename the columns
data.rename(columns=column_rename_map, inplace=True)

# Print value counts for a specific question
print(data['Q10_help_impact_self_view'].value_counts())

# Plot histograms for selected questions
data[['Q1_satisfaction_combined', 'Q2_quality_of_life_combined', 'Q3a_control_over_daily_life']].hist(bins=15,
                                                                                                      figsize=(15, 10))
plt.suptitle('Histograms of Selected Questions')
plt.tight_layout()
plt.show()

# Generate box plots for the same questions
data[['Q1_satisfaction_combined', 'Q2_quality_of_life_combined', 'Q3a_control_over_daily_life']].boxplot(figsize=(8, 6))
plt.title('Box Plots of Selected Questions')
plt.tight_layout()
plt.show()

# Histogram for 'Question 1 combined' with custom settings
plt.hist(data['Q1_satisfaction_combined'],
         bins=range(int(data['Q1_satisfaction_combined'].min()), int(data['Q1_satisfaction_combined'].max()) + 2),
         edgecolor='black')
plt.title('Histogram of Responses for Question 1 Combined')
plt.xlabel('Response Category')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Create a bar chart for the frequency of responses in 'Question 2 combined'
category_counts = data['Q2_quality_of_life_combined'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
plt.bar(category_counts.index, category_counts.values, color='blue', alpha=0.7)
plt.title('Distribution of Responses for Question 2 Combined')
plt.xlabel('Response Category')
plt.ylabel('Frequency')
plt.xticks(ticks=[1, 2, 3, 4, 5], labels=[
    'So good, it could not be better or very good',
    'Good',
    'Alright',
    'Bad',
    'Very bad or So bad, it could not be worse'
])
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Correlation analysis including 'Question 2 combined' using Spearman's rank correlation
numeric_data = data.select_dtypes(include=[np.number])
spearman_corr = numeric_data.corr(method='spearman')
print("Spearman's Rank Correlation Matrix:\n", spearman_corr)

# Extract correlations for 'Question 2 combined' only
q2_corr = spearman_corr['Q2_quality_of_life_combined']

print(q2_corr)

# Create a bar plot for the correlation values of 'Question 2 combined' with all other questions
plt.figure(figsize=(10, 8))
q2_corr.drop('Q2_quality_of_life_combined', inplace=True)  # Drop self-correlation
sns.barplot(x=q2_corr.values, y=q2_corr.index, palette='coolwarm')
plt.title('Correlation of Question 2 Combined with Other Questions')
plt.xlabel('Spearman Correlation Coefficient')
plt.tight_layout()
plt.show()

# Select predictors - focusing on the other survey question responses as potential predictors
question_columns = [col for col in data.columns if
                    # col.startswith('Q') and
                    col != 'Q2_quality_of_life_combined' and col != 'CASSR_code']
print(question_columns)
X = data[question_columns]  # predictor variables
y = data['Q2_quality_of_life_combined']  # target variable

print(np.asarray(X))
# Fit an ordinal logistic regression model without adding a constant
mod = OrderedModel(y, X, distr='logit')
res = mod.fit(method='bfgs', maxiter=500)

print(res.summary())

# # repeat the regression but with just age group, q1, 3a, 9a, 14, 15b
# question_columns = ['Age_group', 'Q1_satisfaction_combined', 'Q3a_control_over_daily_life', 'Q9a_spend_time',
#                     'Q14_general_health', 'Q15b_anxiety_or_depression']
# X = data[question_columns]  # predictor variables
# y = data['Q2_quality_of_life_combined']  # target variable
#
# # Fit an ordinal logistic regression model without adding a constant
# mod = OrderedModel(y, X, distr='logit')
# res = mod.fit(method='bfgs', maxiter=500)
#
# print(res.summary())


# Define the gender groups
group1 = data[data['Gender'] == 1]['Q2_quality_of_life_combined'].dropna()  # Males
group2 = data[data['Gender'] == 2]['Q2_quality_of_life_combined'].dropna()  # Females

# Perform the Mann-Whitney U test
stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
# Conclusion
if p < 0.05:
    print("There is a significant difference in the quality of life ratings between males and females.")
else:
    print("There is no significant difference in the quality of life ratings between males and females.")

# Print the results
print(f"Mann-Whitney U test statistic: {stat}")
print(f"P-value: {p}")

# Assuming 'Gender' and 'Q2_quality_of_life_combined' are appropriately coded
cross_tab = pd.crosstab(data['Q2_quality_of_life_combined'], data['Gender'])
chi2, p_value, dof, expected = chi2_contingency(cross_tab)

print("Chi-Square Test")
print("Chi2 Statistic:", chi2)
print("Degrees of Freedom:", dof)
print("P-value:", p_value)

from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(X)

from sklearn.decomposition import PCA

# Initialize PCA model
pca = PCA(n_components=None)  # n_components=None for full decomposition
principal_components = pca.fit_transform(features_scaled)

# Explained variance ratio by each principal component
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5, align='center',
        label='Individual explained variance')
plt.step(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(cumulative_variance >= 0.95) + 1  # Plus 1 because index is 0-based
print("Number of components to explain 95% variance:", num_components)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Variance')
plt.title('Scree Plot')
plt.axvline(x=num_components, color='r', linestyle='--', label=f'{num_components} components for 95% variance')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Variance')
plt.title('Scree Plot')
plt.axvline(x=num_components, color='r', linestyle='--', label=f'{num_components} components for 95% variance')
plt.legend()
plt.show()

# Create a DataFrame for PCA loadings
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i + 1}' for i in range(len(principal_components[0]))],
                        index=question_columns)

# Plot the square of loadings for each variable on the first few components
loadings_squared = loadings ** 2
sns.heatmap(loadings_squared.iloc[:, :num_components], annot=True, cmap="viridis")
plt.title('Squared Loadings of Variables on Principal Components')
plt.show()

# Identify the variable with the least impact
least_impact_var = loadings_squared.sum(axis=1).idxmin()
print("Variable with the least impact:", least_impact_var)

# Plotting the PCA loadings
plt.figure(figsize=(12, 8))
sns.heatmap(loadings.iloc[:, :10], annot=True, cmap="coolwarm",
            center=0)  # Focus on the first 10 components for clarity
plt.title('Loadings Plot for the First 10 Principal Components')
plt.show()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming data is already loaded and cleaned
X = data.select_dtypes(include=[np.number])  # select numeric columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Standardize the data

# Apply PCA
pca = PCA(n_components=0.85)  # Adjust n_components to the level of explained variance you are comfortable with
X_pca = pca.fit_transform(X_scaled)

# Check the number of components used
print(f"Number of components: {pca.n_components_}")
print(f"Explained variance by each component: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {pca.explained_variance_ratio_.cumsum()}")

# Optionally, review the components to decide on further exclusions
components = pd.DataFrame(pca.components_, columns=X.columns)
print(components.abs().sum(axis=0).sort_values())  # Sum of absolute loadings for each variable



import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


# Filter data for residential care
residential_data = data[data['Support_Setting'] == 2]

# Select relevant columns for clustering
# Example: Using some of the question responses
columns_for_clustering = ['Q1_satisfaction_combined', 'Q2_quality_of_life_combined', 'Q3a_control_over_daily_life', 'Q14_general_health']
cluster_data = residential_data[columns_for_clustering]

# Drop any rows with missing data to avoid issues during clustering
cluster_data.dropna(inplace=True)

# Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_data)

# Perform hierarchical clustering
linked = linkage(cluster_scaled, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           labels=cluster_data.index.tolist(),
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrogram for Residential Care')
plt.xlabel('Index of Entry')
plt.ylabel('Distance')
plt.show()


# Mapping dictionaries based on provided metadata
age_group_mapping = {1: '18-64', 2: '65 and over', 99: 'Value suppressed'}
ethnicity_mapping = {1: 'White', 2: 'Non-White', 3: 'Refused/Prefer not to say /Not-stated', 99: 'Value suppressed'}
gender_mapping = {1: 'Male', 2: 'Female', 3: 'Other', 99: 'Value suppressed'}
support_setting_mapping = {1: 'Community', 2: 'Residential Care', 3: 'Nursing Care', 99: 'Value suppressed'}
stratum_mapping = {1: 'LD', 2: '18-64', 3: '65+Resi Care', 4: '65+Comm.', 99: 'Value suppressed'}
psr_mapping = {1: 'Physical Support', 2: 'Sensory Support', 3: 'Support with Memory and Cognition',
               4: 'Learning Disability Support', 5: 'Mental Health Support', 6: 'Social Support',
               99: 'Value suppressed'}

# Apply mappings to the data
data['Age_group'] = data['Age_group'].map(age_group_mapping)
data['Ethnicity'] = data['Ethnicity'].map(ethnicity_mapping)
data['Gender'] = data['Gender'].map(gender_mapping)
data['Support_Setting'] = data['Support_Setting'].map(support_setting_mapping)
data['Stratum'] = data['Stratum'].map(stratum_mapping)
data['Primary_Support_Reason'] = data['Primary_Support_Reason'].map(psr_mapping)


# Function to create box plots for 'Q2_quality_of_life_combined' across different categories
def plot_box_q2_across_categories(data, category, title):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=category, y='Q2_quality_of_life_combined', data=data)
    plt.title(f'Q2 Quality of Life Responses by {title}')
    plt.xlabel(title)
    plt.ylabel('Q2 Quality of Life Response')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_bar_q2_across_categories(data, category, title):
    plt.figure(figsize=(14, 8))
    ax = sns.countplot(x=category, hue='Q2_quality_of_life_combined', data=data, palette='viridis')
    plt.title(f'Counts of Q2 Quality of Life Responses by {title}')
    plt.xlabel(title)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Q2 Quality of Life', loc='upper right')

    # Adding data labels
    for p in ax.patches:
        count = int(p.get_height())
        if count > 0:  # Ensure we do not annotate empty bars
            ax.annotate(f'{count}', (p.get_x() + p.get_width() / 2., count),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.tight_layout()
    plt.show()


# Function to create summary statistics for 'Q2_quality_of_life_combined' across different categories
def summary_stats_q2_across_categories(data, category):
    summary = data.groupby(category)['Q2_quality_of_life_combined'].describe()
    print(f'Summary Statistics for Q2 Quality of Life Responses by {category}')
    print(summary)
    print('\n')


# Categories to analyze
categories = ['Age_group', 'Ethnicity', 'Gender', 'Support_Setting', 'Stratum', 'Primary_Support_Reason']

# Plot and summary statistics for each category
for category in categories:
    plot_box_q2_across_categories(data, category, category.replace('_', ' ').title())
    plot_bar_q2_across_categories(data, category, category.replace('_', ' ').title())
    summary_stats_q2_across_categories(data, category)
