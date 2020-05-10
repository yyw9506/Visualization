'''
Author: Yuyao Wang
Date: 2020-2-27 / 2020-3-18
'''

# flask
from flask import Flask, request, jsonify
from flask import redirect, render_template

# pandas, numpy, json
import pandas as pd
import numpy as np
import json

# matplotlib
import matplotlib.pyplot as plt

# from kneed import KneeLocator

# sklearn
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.manifold import MDS

app = Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")

'''
elbow for k-means
'''
@app.route('/runElbowOnKmeans')
def runElbowOnKmeans():
    # load dataset
    dataset =  pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    # sse dic and list
    sse = {}
    # list for generating the line plot
    SSE = []

    for index in range(1,10):
        kmeans_model = KMeans(n_clusters=index, max_iter=1000).fit(dataset)
        dataset["clusters"] = kmeans_model.labels_
        sse[index] = kmeans_model.inertia_

    for index in range(1,10):
        SSE.append(sse[index])

    elbow_data = []
    elbow_data = pd.DataFrame(data=elbow_data, columns=["x", "y"])
    elbow_data["x"] = list(sse.keys())
    elbow_data["y"] = list(sse.values())
    elbow_data = elbow_data.to_dict(orient='records')
    elbow_data = {'data': elbow_data}

    print("Elbow Data:", elbow_data)
    return jsonify(elbow_data)

    # print(SSE)
    # X = range(1, 15)
    # plt.xlabel('k for K-means')
    # plt.ylabel('SSE')
    # plt.plot(X, SSE, 'o-')
    # plt.show()

    # print(dataset.keys())
    # print(dataset.index))
    # return jsonify(elbow_data)
    # for row in dataset:
    #     print(row.data)

'''
Scree Plot for PCA (Random Sampling)
'''
@app.route('/randomSamplingPCA_ScreePlot')
def randomSamplingPCA_ScreePlot():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    # random select 25% of the dataset, no repeat data point
    dataset_part = dataset.sample(frac=0.25, random_state=None)
    # print("Length after random sampling: ",dataset_part)

    scaled_data = StandardScaler().fit_transform(dataset)
    scaled_data_part = StandardScaler().fit_transform(dataset_part)
    # print("Scaled data (25%)", type(scaled_data_part))

    pca = PCA(n_components=10)
    pca_data = pca.fit_transform(scaled_data_part)

    scree_plt_data = []
    scree_plt_data = pd.DataFrame(data=scree_plt_data, columns=["x", "y"])
    scree_plt_data["x"] = list(range(1, 11))
    scree_plt_data["y"] = list(pca.explained_variance_ratio_)
    scree_plt_data = scree_plt_data.to_dict(orient='records')
    scree_plt_data = {'data': scree_plt_data}
    return jsonify(scree_plt_data)

'''
Scatter Plot for PCA (Random Sampling)
'''
@app.route('/randomSamplingPCA_ScatterPlot')
def randomSamplingPCA_ScatterPlot():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    # random select 25% of the dataset, no repeat data point
    dataset_part = dataset.sample(frac=0.25, random_state=None)
    scaled_data = StandardScaler().fit_transform(dataset_part)
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    scatter_plt_data = pd.DataFrame(data=pca_data, columns=['x', 'y'])
    scatter_plt_data = scatter_plt_data.to_dict(orient='records')
    scatter_plt_data = {'data': scatter_plt_data}
    return jsonify(scatter_plt_data)

'''
Scree Plot for MDS (Random Sampling)
'''
@app.route("/randomSamplingMDS_ScreePlot")
def randomSamplingMDS_ScreePlot():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    # random select 25% of the dataset, no repeat data point
    dataset_part = dataset.sample(frac=0.25, random_state=None)
    # print("Length after random sampling: ",dataset_part)
    scaled_data_part = StandardScaler().fit_transform(dataset_part)
    stress = []
    for index in range(1, 11):
        euclidean_mds = MDS(n_components=index, dissimilarity='euclidean')
        euclidean_mds.fit_transform(scaled_data_part)
        stress.append(euclidean_mds.stress_)

    scree_plt_data = []
    scree_plt_data = pd.DataFrame(data=scree_plt_data, columns=["x", "y"])
    scree_plt_data["x"] = list(range(1, 11))
    scree_plt_data["y"] = stress
    # print(scree_plt_data)
    scree_plt_data = scree_plt_data.to_dict(orient='records')
    scree_plt_data = {'data': scree_plt_data}
    return jsonify(scree_plt_data)

'''
Scatter Plot for MDS (Random Sampling)
'''
@app.route("/randomSamplingMDS_ScatterPlot")
def randomSamplingMDS_ScatterPlot():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    # random select 25% of the dataset, no repeat data point
    dataset_part = dataset.sample(frac=0.25, random_state=None)
    # print("Length after random sampling: ",dataset_part)
    scaled_data_part = StandardScaler().fit_transform(dataset_part)
    euclidean_mds = MDS(n_components=2, dissimilarity='euclidean')

    scatter_plt_data = pd.DataFrame(data=euclidean_mds.fit_transform(scaled_data_part), columns=['x', 'y'])
    scatter_plt_data = scatter_plt_data.to_dict(orient='records')
    scatter_plt_data = {'data': scatter_plt_data}
    return jsonify(scatter_plt_data)

'''
Scatter Plot for MDS Correlation (random Sampling)
'''
@app.route('/randomSamplingMDSCorr_ScatterPlot')
def randomSamplingMDSCorr_ScatterPlot():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    # random select 25% of the dataset, no repeat data point
    dataset_part = dataset.sample(frac=0.25, random_state=None)
    # print("Length after random sampling: ",dataset_part)
    scaled_data_part = StandardScaler().fit_transform(dataset_part)
    distance_matrix = metrics.pairwise_distances(scaled_data_part, metric='correlation')

    correlation_mds = MDS(n_components=2, dissimilarity='precomputed')
    correlation_mds_data = correlation_mds.fit_transform(distance_matrix)

    correlation_mds_data = pd.DataFrame(data=correlation_mds_data, columns=['x', 'y'])
    correlation_mds_data = correlation_mds_data.to_dict(orient='records')
    correlation_mds_data = {'data': correlation_mds_data}
    return jsonify(correlation_mds_data)

'''
Scree Plot for PCA (Stratified Sampling)
'''
@app.route('/stratifiedSamplingPCA_ScreePlot')
def stratifiedSamplingPCA_ScreePlot():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    kmeans_model = KMeans(n_clusters=4)
    kmeans_model = kmeans_model.fit(dataset)
    dataset["clusters"] = kmeans_model.labels_
    # print(dataset)

    cluster_data_0 = dataset[dataset["clusters"] == 0].sample(frac=0.25, random_state=None)
    cluster_data_1 = dataset[dataset["clusters"] == 1].sample(frac=0.25, random_state=None)
    cluster_data_2 = dataset[dataset["clusters"] == 2].sample(frac=0.25, random_state=None)
    cluster_data_3 = dataset[dataset["clusters"] == 3].sample(frac=0.25, random_state=None)

    stratified_sampled_data = []
    data_frame = pd.DataFrame(stratified_sampled_data,
                              columns=['radius_mean', 'texture_mean', 'perimeter_mean',
                                       'area_mean', 'smoothness_mean', 'compactness_mean',
                                       'concavity_mean', 'concave points_mean', 'symmetry_mean',
                                       'fractal_dimension_mean', 'clusters'])
    stratified_sampled_data = data_frame.append(cluster_data_0, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_1, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_2, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_3, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.drop(['clusters'], axis=1)

    scaled_data = StandardScaler().fit_transform(stratified_sampled_data)
    pca = PCA(n_components=10)
    pca_data = pca.fit_transform(scaled_data)

    scree_plt_data = []
    scree_plt_data = pd.DataFrame(data=scree_plt_data, columns=["x", "y"])
    scree_plt_data["x"] = list(range(1, 11))
    scree_plt_data["y"] = list(pca.explained_variance_ratio_)
    scree_plt_data = scree_plt_data.to_dict(orient='records')
    scree_plt_data = {'data': scree_plt_data}
    return jsonify(scree_plt_data)

'''
Scatter Plot for PCA (Stratified Sampling)
'''
@app.route('/stratifiedSamplingPCA_ScatterPlot')
def stratifiedSamplingPCA_ScatterPlot():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    kmeans_model = KMeans(n_clusters=4)
    kmeans_model = kmeans_model.fit(dataset)
    dataset["clusters"] = kmeans_model.labels_
    # print(kmeans_model.labels_)
    # print(dataset)

    cluster_data_0 = dataset[dataset["clusters"] == 0].sample(frac=0.25, random_state=None)
    cluster_data_1 = dataset[dataset["clusters"] == 1].sample(frac=0.25, random_state=None)
    cluster_data_2 = dataset[dataset["clusters"] == 2].sample(frac=0.25, random_state=None)
    cluster_data_3 = dataset[dataset["clusters"] == 3].sample(frac=0.25, random_state=None)

    stratified_sampled_data = []
    data_frame = pd.DataFrame(stratified_sampled_data,
                         columns=['radius_mean', 'texture_mean', 'perimeter_mean',
                                  'area_mean', 'smoothness_mean', 'compactness_mean',
                                  'concavity_mean', 'concave points_mean', 'symmetry_mean',
                                  'fractal_dimension_mean', 'clusters'])
    stratified_sampled_data = data_frame.append(cluster_data_0, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_1, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_2, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_3, ignore_index=True)

    clusters_columns = stratified_sampled_data["clusters"]
    stratified_sampled_data = stratified_sampled_data.drop(['clusters'], axis=1)
    scaled_data = StandardScaler().fit_transform(stratified_sampled_data)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    pca_data = np.append(pca_data, clusters_columns.values.reshape(142, 1), axis=1)
    scatter_plt_data = pd.DataFrame(data=pca_data, columns=['x', 'y', 'cluster'])
    scatter_plt_data = scatter_plt_data.to_dict(orient='records')
    scatter_plt_data = {'data': scatter_plt_data}
    return jsonify(scatter_plt_data)

'''
Scree Plot for MDS (Stratified Sampling)
'''
@app.route('/stratifiedSamplingMDS_ScreePlot')
def stratifiedSamplingMDS_ScreePlot():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    kmeans_model = KMeans(n_clusters=4)
    kmeans_model = kmeans_model.fit(dataset)
    dataset["clusters"] = kmeans_model.labels_
    # print(dataset)

    cluster_data_0 = dataset[dataset["clusters"] == 0].sample(frac=0.25, random_state=None)
    cluster_data_1 = dataset[dataset["clusters"] == 1].sample(frac=0.25, random_state=None)
    cluster_data_2 = dataset[dataset["clusters"] == 2].sample(frac=0.25, random_state=None)
    cluster_data_3 = dataset[dataset["clusters"] == 3].sample(frac=0.25, random_state=None)

    stratified_sampled_data = []
    data_frame = pd.DataFrame(stratified_sampled_data,
                              columns=['radius_mean', 'texture_mean', 'perimeter_mean',
                                       'area_mean', 'smoothness_mean', 'compactness_mean',
                                       'concavity_mean', 'concave points_mean', 'symmetry_mean',
                                       'fractal_dimension_mean', 'clusters'])
    stratified_sampled_data = data_frame.append(cluster_data_0, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_1, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_2, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_3, ignore_index=True)

    stratified_sampled_data = stratified_sampled_data.drop(['clusters'], axis=1)
    scaled_data = StandardScaler().fit_transform(stratified_sampled_data)
    stress = []

    for index in range(1,11):
        euclidean_mds = MDS(n_components=index, dissimilarity='euclidean')
        euclidean_mds_data = euclidean_mds.fit_transform(scaled_data)
        stress.append(euclidean_mds.stress_)

    scree_plt_data = []
    scree_plt_data = pd.DataFrame(data=scree_plt_data, columns=["x", "y"])
    scree_plt_data["x"] = list(range(1, 11))
    scree_plt_data["y"] = stress
    scree_plt_data = scree_plt_data.to_dict(orient='records')
    scree_plt_data = {'data': scree_plt_data}
    return jsonify(scree_plt_data)

'''
Scatter Plot for MDS (Stratified Sampling)
'''
@app.route('/stratifiedSamplingMDS_ScatterPlot')
def stratifiedSamplingMDS_ScatterPlot():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    kmeans_model = KMeans(n_clusters=4)
    kmeans_model = kmeans_model.fit(dataset)
    dataset["clusters"] = kmeans_model.labels_
    # print(dataset)

    cluster_data_0 = dataset[dataset["clusters"] == 0].sample(frac=0.25, random_state=None)
    cluster_data_1 = dataset[dataset["clusters"] == 1].sample(frac=0.25, random_state=None)
    cluster_data_2 = dataset[dataset["clusters"] == 2].sample(frac=0.25, random_state=None)
    cluster_data_3 = dataset[dataset["clusters"] == 3].sample(frac=0.25, random_state=None)

    stratified_sampled_data = []
    data_frame = pd.DataFrame(stratified_sampled_data,
                              columns=['radius_mean', 'texture_mean', 'perimeter_mean',
                                       'area_mean', 'smoothness_mean', 'compactness_mean',
                                       'concavity_mean', 'concave points_mean', 'symmetry_mean',
                                       'fractal_dimension_mean', 'clusters'])
    stratified_sampled_data = data_frame.append(cluster_data_0, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_1, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_2, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_3, ignore_index=True)

    clusters_columns = stratified_sampled_data["clusters"]
    stratified_sampled_data = stratified_sampled_data.drop(['clusters'], axis=1)
    scaled_data = StandardScaler().fit_transform(stratified_sampled_data)

    euclidean_mds = MDS(n_components=2, dissimilarity='euclidean')
    euclidean_mds_data = euclidean_mds.fit_transform(scaled_data)
    euclidean_mds_data=np.append(euclidean_mds_data, clusters_columns.values.reshape(142,1), axis=1)
    scatter_plt_data = pd.DataFrame(data=euclidean_mds_data,columns=['x', 'y','cluster'])
    scatter_plt_data = scatter_plt_data.to_dict(orient='records')
    scatter_plt_data = {'data': scatter_plt_data}
    # print(scatter_plt_data)
    return jsonify(scatter_plt_data)

'''
Scatter Plot for MDS Correlation (Stratified Sampling)
'''
@app.route('/stratifiedSamplingMDSCorr_ScatterPlot')
def stratifiedSamplingMDSCorr_ScatterPlot():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    kmeans_model = KMeans(n_clusters=4)
    kmeans_model = kmeans_model.fit(dataset)
    dataset["clusters"] = kmeans_model.labels_
    # print(dataset)

    cluster_data_0 = dataset[dataset["clusters"] == 0].sample(frac=0.25, random_state=None)
    cluster_data_1 = dataset[dataset["clusters"] == 1].sample(frac=0.25, random_state=None)
    cluster_data_2 = dataset[dataset["clusters"] == 2].sample(frac=0.25, random_state=None)
    cluster_data_3 = dataset[dataset["clusters"] == 3].sample(frac=0.25, random_state=None)

    stratified_sampled_data = []
    data_frame = pd.DataFrame(stratified_sampled_data,
                              columns=['radius_mean', 'texture_mean', 'perimeter_mean',
                                       'area_mean', 'smoothness_mean', 'compactness_mean',
                                       'concavity_mean', 'concave points_mean', 'symmetry_mean',
                                       'fractal_dimension_mean', 'clusters'])
    stratified_sampled_data = data_frame.append(cluster_data_0, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_1, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_2, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_3, ignore_index=True)

    clusters_columns = stratified_sampled_data["clusters"]
    stratified_sampled_data = stratified_sampled_data.drop(['clusters'], axis=1)
    scaled_data = StandardScaler().fit_transform(stratified_sampled_data)

    distance_matrix = metrics.pairwise_distances(scaled_data, metric='correlation')
    correlation_mds = MDS(n_components=2, dissimilarity='precomputed')
    correlation_mds_data = correlation_mds.fit_transform(distance_matrix)
    correlation_mds_data = np.append(correlation_mds_data, clusters_columns.values.reshape(142, 1), axis=1)
    scatter_plt_data = pd.DataFrame(data=correlation_mds_data, columns=['x', 'y', 'cluster'])
    scatter_plt_data = scatter_plt_data.to_dict(orient='records')
    scatter_plt_data = {'data': scatter_plt_data}
    # print(scatter_plt_data)
    return jsonify(scatter_plt_data)

'''
Scree Plot for PCA (Original)
'''
@app.route('/originalPCA_ScreePlot')
def originalPCA_ScreePlot():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)
    # print("Length after random sampling: ",dataset_part)

    scaled_data = StandardScaler().fit_transform(dataset)
    # print("Scaled data (25%)", type(scaled_data_part))

    pca = PCA(n_components=10)
    pca_data = pca.fit_transform(scaled_data)

    scree_plt_data = []
    scree_plt_data = pd.DataFrame(data=scree_plt_data, columns=["x", "y"])
    scree_plt_data["x"] = list(range(1, 11))
    scree_plt_data["y"] = list(pca.explained_variance_ratio_)
    scree_plt_data = scree_plt_data.to_dict(orient='records')
    scree_plt_data = {'data': scree_plt_data}
    return jsonify(scree_plt_data)

'''
Scatter Plot for PCA (Original)
'''
@app.route('/originalPCA_ScatterPlot')
def originalPCA_ScatterPlot():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    scaled_data = StandardScaler().fit_transform(dataset)
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    scatter_plt_data = pd.DataFrame(data=pca_data, columns=['x', 'y'])
    scatter_plt_data = scatter_plt_data.to_dict(orient='records')
    scatter_plt_data = {'data': scatter_plt_data}
    return jsonify(scatter_plt_data)

'''
Scree Plot for MDS (Original)
'''
@app.route("/originalMDS_ScreePlot")
def originalMDS_ScreePlot():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    # print("Length after random sampling: ",dataset_part)
    scaled_data= StandardScaler().fit_transform(dataset)
    stress = []
    for index in range(1, 11):
        print("round:",index)
        euclidean_mds = MDS(n_components=index, dissimilarity='euclidean')
        euclidean_mds.fit_transform(scaled_data)
        stress.append(euclidean_mds.stress_)

    scree_plt_data = []
    scree_plt_data = pd.DataFrame(data=scree_plt_data, columns=["x", "y"])
    scree_plt_data["x"] = list(range(1, 11))
    scree_plt_data["y"] = stress
    # print(scree_plt_data)
    scree_plt_data = scree_plt_data.to_dict(orient='records')
    scree_plt_data = {'data': scree_plt_data}
    return jsonify(scree_plt_data)

'''
Scatter Plot for MDS (Original)
'''
@app.route("/originalMDS_ScatterPlot")
def originalMDS_ScatterPlot():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    # print("Length after random sampling: ",dataset_part)
    scaled_data_part = StandardScaler().fit_transform(dataset)
    euclidean_mds = MDS(n_components=2, dissimilarity='euclidean')

    scatter_plt_data = pd.DataFrame(data=euclidean_mds.fit_transform(scaled_data_part), columns=['x', 'y'])
    scatter_plt_data = scatter_plt_data.to_dict(orient='records')
    scatter_plt_data = {'data': scatter_plt_data}
    return jsonify(scatter_plt_data)

'''
Scatter Plot for MDS Correlation (Original)
'''
@app.route('/originalMDSCorr_ScatterPlot')
def originalMDSCorr_ScatterPlot():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    # print("Length after random sampling: ",dataset_part)
    scaled_data_part = StandardScaler().fit_transform(dataset)
    distance_matrix = metrics.pairwise_distances(scaled_data_part, metric='correlation')

    correlation_mds = MDS(n_components=2, dissimilarity='precomputed')
    correlation_mds_data = correlation_mds.fit_transform(distance_matrix)

    correlation_mds_data = pd.DataFrame(data=correlation_mds_data, columns=['x', 'y'])
    correlation_mds_data = correlation_mds_data.to_dict(orient='records')
    correlation_mds_data = {'data': correlation_mds_data}
    return jsonify(correlation_mds_data)

'''
Original Scatter PlotMatrix
'''
@app.route('/originalScatterPlotMatrix')
def originalScatterPlotMatrix():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    # drop other columns
    dataset = dataset.drop(['radius_mean'], axis=1)
    dataset = dataset.drop(['perimeter_mean'], axis=1)
    dataset = dataset.drop(['area_mean'], axis=1)
    dataset = dataset.drop(['smoothness_mean'], axis=1)
    dataset = dataset.drop(['compactness_mean'], axis=1)
    dataset = dataset.drop(['concavity_mean'], axis=1)
    dataset = dataset.drop(['concave points_mean'], axis=1)

    scatter_plt_matrix_data = dataset.to_dict(orient='records')
    scatter_plt_matrix_data = {'data': scatter_plt_matrix_data}
    return jsonify(scatter_plt_matrix_data)

'''
Random Sampling Scatter PlotMatrix
'''
@app.route('/randomSamplingScatterPlotMatrix')
def randomSamplingScatterPlotMatrix():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    dataset = dataset.sample(frac=0.25, random_state=None)
    scaled_data = StandardScaler().fit_transform(dataset)
    pca = PCA(n_components=4)
    pca_data = pca.fit_transform(scaled_data)
    # print(pca.components_.T)

    dataset_columns = dataset.keys();

    print(dataset_columns)
    pca_columns = pd.DataFrame(data=pca.components_.T, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4'])
    # print(pca_columns);
    pca_columns.insert(loc=0, column="Attribute", value=list(dataset))
    pca_columns['SSL'] = pca_columns.drop(['Attribute'], axis=1).apply(np.square).sum(axis=1)
    print(pca_columns)

    sorted_pca_columns = sorted(pca_columns['SSL'], reverse=True)[0:3]
    PCA_columns = []
    PCA_columns_left = []
    # print(sorted_pca_columns)
    print("-------------------------------------------")
    print("Three attributes with highest PCA Loadings.")
    for index in range(0, 3):
        # print(pca_columns[pca_columns.get('SSL') == sorted_pca_columns[index]]["Attribute"].to_dict())
        PCA_columns.append(pca_columns[pca_columns.get('SSL') == sorted_pca_columns[index]]["Attribute"].to_dict());

    for element in PCA_columns:
        for index in range(0, 10):
            if (element.get(index)):
                print(element.get(index))
                PCA_columns_left.append(element.get(index))

    print(PCA_columns_left)
    print("-------------------------------------------")
    print("Columns to be dropped.")
    for column in dataset_columns:
        if column not in PCA_columns_left:
            dataset = dataset.drop([column], axis=1)

    scatter_plt_matrix_data = dataset.to_dict(orient='records')
    scatter_plt_matrix_data = {'data': scatter_plt_matrix_data}
    return jsonify(scatter_plt_matrix_data)

'''
Stratified Sampling Scatter Plot Matrix
'''
@app.route("/stratifiedSamplingSatterPlotMatrix")
def stratifiedSamplingSatterPlotMatrix():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    kmeans_model = KMeans(n_clusters=4)
    kmeans_model = kmeans_model.fit(dataset)
    dataset["clusters"] = kmeans_model.labels_
    # print(dataset)

    cluster_data_0 = dataset[dataset["clusters"] == 0].sample(frac=0.25, random_state=None)
    cluster_data_1 = dataset[dataset["clusters"] == 1].sample(frac=0.25, random_state=None)
    cluster_data_2 = dataset[dataset["clusters"] == 2].sample(frac=0.25, random_state=None)
    cluster_data_3 = dataset[dataset["clusters"] == 3].sample(frac=0.25, random_state=None)

    dataset_columns = dataset.keys();

    stratified_sampled_data = []
    data_frame = pd.DataFrame(stratified_sampled_data,
                              columns=['radius_mean', 'texture_mean', 'perimeter_mean',
                                       'area_mean', 'smoothness_mean', 'compactness_mean',
                                       'concavity_mean', 'concave points_mean', 'symmetry_mean',
                                       'fractal_dimension_mean', 'clusters'])

    stratified_sampled_data = data_frame.append(cluster_data_0, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_1, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_2, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_3, ignore_index=True)

    clusters_columns = stratified_sampled_data["clusters"]
    stratified_sampled_data = stratified_sampled_data.drop(['clusters'], axis=1)

    scaled_data = StandardScaler().fit_transform(stratified_sampled_data)
    pca = PCA(n_components=4)
    pca_data = pca.fit_transform(scaled_data)
    # print(pca.components_.T)
    print(stratified_sampled_data)
    pca_columns = pd.DataFrame(data=pca.components_.T, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4'])
    # print(pca_columns);
    pca_columns.insert(loc=0, column="Attribute", value=list(stratified_sampled_data))
    pca_columns['SSL'] = pca_columns.drop(['Attribute'], axis=1).apply(np.square).sum(axis=1)
    print(pca_columns)

    sorted_pca_columns = sorted(pca_columns['SSL'], reverse=True)[0:3]
    PCA_columns = []
    PCA_columns_left = []
    # print(sorted_pca_columns)
    print("-------------------------------------------")
    print("Three attributes with highest PCA Loadings.")
    for index in range(0, 3):
        # print(pca_columns[pca_columns.get('SSL') == sorted_pca_columns[index]]["Attribute"].to_dict())
        PCA_columns.append(pca_columns[pca_columns.get('SSL') == sorted_pca_columns[index]]["Attribute"].to_dict());

    for element in PCA_columns:
        for index in range(0, 10):
            if (element.get(index)):
                print(element.get(index))
                PCA_columns_left.append(element.get(index))
    PCA_columns_left.append("clusters")

    print(PCA_columns_left)
    print("-------------------------------------------")
    print("Columns to be dropped.")
    for column in dataset_columns:
        if column not in PCA_columns_left:
            print(column)
            stratified_sampled_data = stratified_sampled_data.drop([column], axis=1)

    stratified_sampled_data.insert(loc=3, column="clusters", value=list(clusters_columns))

    scatter_plt_matrix_data = stratified_sampled_data.to_dict(orient='records')
    scatter_plt_matrix_data = {'data': scatter_plt_matrix_data}
    return jsonify(scatter_plt_matrix_data)

def getPCALoadings():
    # load dataset
    dataset = pd.read_csv("data/breast_cancer_data_mod.csv")

    # drop id and diagnosis
    dataset = dataset.drop(['id'], axis=1)
    dataset = dataset.drop(['diagnosis'], axis=1)

    kmeans_model = KMeans(n_clusters=4)
    kmeans_model = kmeans_model.fit(dataset)
    dataset["clusters"] = kmeans_model.labels_
    # print(dataset)

    cluster_data_0 = dataset[dataset["clusters"] == 0].sample(frac=0.25, random_state=None)
    cluster_data_1 = dataset[dataset["clusters"] == 1].sample(frac=0.25, random_state=None)
    cluster_data_2 = dataset[dataset["clusters"] == 2].sample(frac=0.25, random_state=None)
    cluster_data_3 = dataset[dataset["clusters"] == 3].sample(frac=0.25, random_state=None)

    dataset_columns = dataset.keys();

    stratified_sampled_data = []
    data_frame = pd.DataFrame(stratified_sampled_data,
                              columns=['radius_mean', 'texture_mean', 'perimeter_mean',
                                       'area_mean', 'smoothness_mean', 'compactness_mean',
                                       'concavity_mean', 'concave points_mean', 'symmetry_mean',
                                       'fractal_dimension_mean', 'clusters'])

    stratified_sampled_data = data_frame.append(cluster_data_0, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_1, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_2, ignore_index=True)
    stratified_sampled_data = stratified_sampled_data.append(cluster_data_3, ignore_index=True)

    clusters_columns = stratified_sampled_data["clusters"]
    stratified_sampled_data = stratified_sampled_data.drop(['clusters'], axis=1)

    scaled_data = StandardScaler().fit_transform(stratified_sampled_data)
    pca = PCA(n_components=4)
    pca_data = pca.fit_transform(scaled_data)
    print(pca.components_.T)
    print("----------------------")
    print(stratified_sampled_data)
    pca_columns = pd.DataFrame(data=pca.components_.T, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4'])
    # print(pca_columns);
    pca_columns.insert(loc=0, column="Attribute", value=list(stratified_sampled_data))
    pca_columns['SSL'] = pca_columns.drop(['Attribute'], axis=1).apply(np.square).sum(axis=1)
    print(pca_columns)

    sorted_pca_columns = sorted(pca_columns['SSL'], reverse=True)[0:3]
    PCA_columns =[]
    PCA_columns_left = []
    # print(sorted_pca_columns)
    print("-------------------------------------------")
    print("Three attributes with highest PCA Loadings.")
    for index in range(0,3):
        print(pca_columns[pca_columns.get('SSL') == sorted_pca_columns[index]])
        PCA_columns.append(pca_columns[pca_columns.get('SSL') == sorted_pca_columns[index]]["Attribute"].to_dict());

    for element in PCA_columns:
        for index in range(0,10):
            if (element.get(index)):
                print(element.get(index))
                PCA_columns_left.append(element.get(index))
    PCA_columns_left.append("clusters")

    print(PCA_columns_left)
    print("-------------------------------------------")
    print("Columns to be dropped.")
    for column in dataset_columns:
        if column not in PCA_columns_left:
            print(column)
            stratified_sampled_data = stratified_sampled_data.drop([column], axis=1)

    stratified_sampled_data.insert(loc=3, column="clusters" , value=list(clusters_columns))

if __name__ == "main":
    print("running...")
    app.run(host='0.0.0.0',port=5000,debug=True)

# getPCALoadings()
# getPCALoadings();
# originalScatterPlotMatrix();
# scatterPlotMatrix();
# randomSamplingScatterPlotMatrix()
# stratifiedSamplingPCA_ScatterPlot()