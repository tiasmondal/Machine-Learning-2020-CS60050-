NMI scores-
1.Agglomerative clustering without PCA- 0.022844902439705304
2.Agglomerative clustering with PCA- 0.03743944772947285
3.Kmeans clustering without PCA- 0.296208546614843
4.Kmeans clustering with PCA - 0.36621238512049664

# For AGGLOMERATIVE CLUSTERING

Steps to be followed-
cd Agglomerative clustering

->For normal agglomerative clustering-

1.Run the "dataset_preprocessing.py" to obtain the normalized tf-idf matrix of the DTM at "data_modified.csv"

2.(Can be skipped)Now run the "distance_matrix_calculation.py" file to get a nXn distance matrix pertaining to e^(-cosine similarity) saved in "distance_new_inverse_cosine.dat"
and used for furthur steps. This method will take a very huge amount of time, and advised be run only once, this step or can be skipped as I have already run this.

3. Run the "agglomerative_clustering.py" file to get the agglomerative clusters to be saved in "clusters_agglomerative.txt" and "clusters_agglomerative.dat" files.

4.For NMI score run "NMI_score.py"

->For agglomerative clustering after reducing dimension to 100 by PCA

1. Run "dataset_formation_pca", this will form the PCA dataset in "AllBooks_baseline_DTM_Labelled_pca.csv" file.

2.Run the "dataset_preprocessing_pca.py" to obtain the normalized tf-idf matrix of the DTM at "data_modified_pca.csv"

3.Now run the "distance_matrix_calculation_pca.py" file to get a nXn distance matrix pertaining to e^(-cosine similarity) saved in "distance_new_inverse_cosine_pca.dat"
and used for furthur steps.This will take lower time as the dimensions are reduced.

4.Run the "agglomerative_clustering_pca.py" file to get the agglomerative clusters to be saved in "clusters_agglomerative_pca.txt" and "clusters_agglomerative_pca.dat" files.

5.For NMI score run "NMI_score.py"


# For K-MEANS CLUSTERING

Steps to be followed-
cd Kmeans clustering

->For normal K-Means clustering

1.Run the "dataset_preprocessing.py" to obtain the normalized tf-idf matrix of the DTM at "data_modified.csv"

2.Run the "Kmeans_clustering.py" file to get the agglomerative clusters to be saved in "clusters_k_means.txt" and "clusters_k_means.dat" files.

3.For NMI score run "NMI_score.py"

->For agglomerative clustering after reducing dimension to 100 by PCA

1. Run "dataset_formation_pca", this will form the PCA dataset in "AllBooks_baseline_DTM_Labelled_pca.csv" file.

2.Run the "dataset_preprocessing_pca.py" to obtain the normalized tf-idf matrix of the DTM at "data_modified_pca.csv"

3. Run the "Kmeans_clustering_pca.py" file to get the agglomerative clusters to be saved in "clusters_k_means_pca.txt" and "clusters_k_means_pca.dat" files.

4.For NMI score run "NMI_score.py"
