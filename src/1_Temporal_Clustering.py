# Use the temporal feature information to generate the clusters for further prediction
# There are two steps:
#   1. Use the Umap distribution to get the initial clusters
#   2. separate the subclusters from the largest initial cluster to achieve the fix number of group user selected
#

from lib.Feature_Collection import *


# Parameters ===========================================================================================================
# Time Periods
time_list = ['041520','043020','051520','053120','061520','063020','071520','073120','081520','083120','091520','093020',
             '101520','103120','111520','113020','121520','123120','011521','013121','021521','022821','031521','033121',
             '041521','043021','051521','053121','061521','063021','071521','073121','081521','083121','091521','093021',
             '101521','103121','111521','113021','121521','123121','011522','013122','021522','022822','031522','033122']

path = 'Data/COVID-19_Dataset_Counties_US.xlsx'
save_path_data_collection = 'Data/Feautre_Collection/'
Save_Path_Clusters = 'Results/Clustering Label/'

# Parameters for UMAP (Fit the distribution)
list_of_n_neighbors = 20
list_of_min_dist = 0

window_size = 7          # Average 7 days
max_cluster_num = 8      # Set the final clustering number
All_Cluster_label = []   # Save all results

# Main =================================================================================================================

if __name__ == '__main__':

    # Feature Collection ===============================================================================================

    data_collection(data_path=path, save_path=save_path_data_collection, time_list=time_list, window_size=window_size)

    # Temporal Clusteting ==============================================================================================

    for i in tqdm(range(len(time_list))):
        Ori_Data_file = save_path_data_collection + 'note_{}_01_C_NewData.mat'.format(time_list[i])
        Data_file = save_path_data_collection + 'note_{}_02_C_New_features.mat'.format(time_list[i])

        Initial_save_path = Initial_UMAP_Distribution(Data_file, Save_Path_Clusters, time_list[i])

        tempdata, initial_label, Num_cluster_need = Initial_Clustering(Data_file, Initial_save_path, Save_Path_Clusters, list_of_n_neighbors,
                                                                       list_of_min_dist, max_cluster_num, time_list[i])

        temp_final_label = subclustering_fun(Save_Path_Clusters, Initial_save_path, initial_label, Num_cluster_need, time_list[i], list_of_min_dist)

        # sort list=====================================================================================================
        # Measure the infection number during last 7 days of the time period and then descend the order
        # In other words, cluster 0 is the highest risk of infection scenario

        Final_label = Reordering_Clusters(Ori_Data_file, Initial_save_path, temp_final_label, list_of_n_neighbors, list_of_min_dist, time_list[i])

        if len(All_Cluster_label) == 0:
            All_Cluster_label = Final_label[:, np.newaxis]
        else:
            All_Cluster_label = np.hstack((All_Cluster_label, Final_label[:, np.newaxis]))
        print(time_list[i] + ' is finished for the Sub-Clustering ============================================')

        time.sleep(0.001)

    # Save all clustering results into orignial xlsx file
    save_to_file(path, time_list, All_Cluster_label)

