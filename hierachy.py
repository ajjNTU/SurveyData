import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, low_memory=False)

    # Filter data based on specific conditions
    data = data[
        (data['Response'] == 1) &
        (data['SupportSetting'] == 2) &
        (data['Stratum'] != 99) &
        (data['Stratum'].notnull()) &
        (data['LaCode'] == '511')
        ]
    print(f"Data shape after filtering: {data.shape}")
    # Define and remove unnecessary columns
    columns_to_drop = [
        'Geography_Code', 'Method of collection', 'Response',
        'An advocate was used to help to complete the questionnaire',
        'An interpreter was used to help to complete the questionnaire',
        'translated', 'Version of the questionnaire used', 'Service user is a replacement in the sample',
        'Responded to the original or reminder', 'Question 1 standard', 'Question 2 standard',
        'Question 1 easy-read', 'Question 2 easy-read',
        'Question 22', 'Question 23a', 'Question 23b', 'Question 23c', 'Question 23d',
        'Question 23e', 'Question 23f'
    ]

    data.drop(columns=columns_to_drop, inplace=True)

    # Remove optional questions
    optional_questions = ['Question 3b', 'Question 2b', 'Question 2c', 'Question 4b',
                          'Question 5b', 'Question 6b', 'Question 8b', 'Question 9b']
    data.drop(columns=optional_questions, inplace=True)

    return data


def impute_with_group_median_mode(data, group_column):
    for column in data.columns:
        if data[column].dtype == 'object' or column == group_column:
            continue
        # Impute missing values using the mode for categorical and median for continuous variables
        if data[column].dtype in ['float64', 'int64']:
            data[column] = data[column].fillna(data.groupby(group_column)[column].transform('median'))
        else:
            mode_impute = lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            data[column] = data[column].fillna(data.groupby(group_column)[column].transform(mode_impute))
    return data


def hierarchical_clustering(data, columns_for_clustering):
    cluster_data = data[columns_for_clustering].dropna()

    # Standardize the data
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_data)

    # Perform hierarchical clustering
    linked = linkage(cluster_scaled, method='ward')
    return linked, cluster_data


def plot_dendrogram(linked, labels, cluster_labels):
    # Define a color map for the clusters
    color_map = {1: 'red', 2: 'green', 3: 'blue', 4: 'orange', 5: 'purple'}

    # Assign colors to the cluster labels
    cluster_colors = [color_map[label] for label in cluster_labels]

    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               labels=labels,
               color_threshold=0,
               above_threshold_color='gray',
               link_color_func=lambda k: cluster_colors[k])

    plt.title('Dendrogram for Residential Care')
    plt.xlabel('Index of Entry')
    plt.ylabel('Distance')
    plt.show()

def analyze_clusters(data, cluster_data, linked, num_clusters):
    cluster_labels = fcluster(linked, num_clusters, criterion='maxclust')
    cluster_data['Cluster'] = cluster_labels

    data = data.join(cluster_data[['Cluster']], how='inner')

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Cluster', y='Question 2 combined', data=data)
    plt.title('Distribution of Q2 Quality of Life Across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Q2 Quality of Life Combined')
    plt.show()


def main():
    data = load_and_preprocess_data('pss-ascs-eng-2022-23-CSV.csv')
    data = impute_with_group_median_mode(data, 'Stratum')

    columns_for_clustering = ['Question 1 combined', 'Question 3a', 'Question 14']
    linked, cluster_data = hierarchical_clustering(data, columns_for_clustering)

    num_clusters = 3  # Specify the desired number of clusters
    cluster_labels = fcluster(linked, num_clusters, criterion='maxclust')

    plot_dendrogram(linked, cluster_data.index.tolist(), cluster_labels)
    analyze_clusters(data, cluster_data, linked, num_clusters)

if __name__ == "__main__":
    main()
