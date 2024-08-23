#Imports
import pandas as pd
import numpy as np
import pickle
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.clustering import silhouette_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path, start_date, end_date, num_columns=30):
    # Load and prepare data
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Filter data for test set
    data_test = data[(data['Date'] >= start_date) & (data['Date'] < end_date)]
       
    # Extract load and PV data
    load_columns = [f"load_{i}" for i in range(1, num_columns + 1)]
    pv_columns = [f"pv_{i}" for i in range(1, num_columns + 1)]
    
    load_test = data_test[load_columns]
    pv_test = data_test[pv_columns]
    
    # Extract the date column
    date_column = data_test['Date'].reset_index(drop=True)
    
    return load_test, pv_test, date_column

def calculate_prosumption(load_test, pv_test, date_column, num_columns=30):
    # Calculate prosumption
    test_prosumption = pd.DataFrame()
    
    for i in range(1, num_columns + 1):
        test_prosumption[f"prosumption_{i}"] = load_test[f"load_{i}"] - pv_test[f"pv_{i}"]
    
    # Add the date column
    test_prosumption.insert(0, "Date", date_column)
    
    # Reset index and transpose the dataframe
    test_prosumption = test_prosumption.reset_index(drop=True)
    
    return test_prosumption

def setup_ausgrid_prosumption_data(file_path, start_date='2010-07-01', end_date='2011-07-01', num_columns=30):
    load_test, pv_test, date_column = load_and_prepare_data(file_path, start_date, end_date, num_columns)
    final_prosumption = calculate_prosumption(load_test, pv_test, date_column, num_columns)
    return final_prosumption

def scaled_mean_over_48_timesteps(data, num_colums=30):
    data.insert(0, 'timeslot', data['Date'].dt.time, True)
    data.drop(columns=['Date'], inplace=True)
    data = data.groupby(['timeslot']).mean()
    data = data.reset_index(drop=True).transpose()
    data = TimeSeriesScalerMinMax().fit_transform(data.values.reshape((num_colums,48,1)))
    return data

def kmeans_clustered_buildings(num_clusters):
    
    #Get cluster array
    cluster_data = []
    if num_clusters == 1:
        cluster_data = np.array([0] * 30)
    elif num_clusters == 30:
        cluster_data = np.array(list(range(0, 31)))
    else:
        cluster_data = pickle.load(open(f"finalclustering/kmeans_mean_timeslot_{num_clusters}.pkl", "rb"))
        cluster_data = cluster_data.labels_ 

    # Convert array to dictionary
    clustered_buildings = {i: [] for i in range(num_clusters)}
    for cluster_number in range(num_clusters):
        buildings_in_cluster = np.where(cluster_data == cluster_number)[0] + 1
        clustered_buildings[cluster_number] = buildings_in_cluster

    return clustered_buildings

def plot_results(results):

    plt.figure(figsize=(12, 6))
    results.plot(subplots=True, layout=(1, 3), figsize=(10, 4), title="Clustering Scores for Different Number of Clusters")
    plt.tight_layout()
    plt.show()

def calculate_clustering_scores(kmeans_data, num_clusters_range, model_path_template="finalclustering/kmeans_mean_timeslot_{}.pkl"):

    # Initialize dictionaries to store scores
    silhouette_scores = {}
    davies_bouldin_scores = {}
    calinski_harabasz_scores = {}

    # Flatten the data for Davies-Bouldin and Calinski-Harabasz scores
    flat_data = kmeans_data.reshape((kmeans_data.shape[0], -1))

    # Calculate scores for each number of clusters
    for i in num_clusters_range:
        # Load the KMeans model for the current number of clusters
        kmeans_cluster = pickle.load(open(model_path_template.format(i), "rb"))
        
        silhouette_scores[i] = silhouette_score(kmeans_data, kmeans_cluster.labels_, metric="dtw", n_jobs=-1)
        davies_bouldin_scores[i] = davies_bouldin_score(flat_data, kmeans_cluster.labels_)
        calinski_harabasz_scores[i] = calinski_harabasz_score(flat_data, kmeans_cluster.labels_)

    # Create a DataFrame to store the results
    results = pd.DataFrame({
        'Silhouette Score': silhouette_scores,
        'Davies-Bouldin Score': davies_bouldin_scores,
        'Calinski-Harabasz Score': calinski_harabasz_scores
    })

    return results