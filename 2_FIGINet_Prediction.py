# Use Bi-SLSTM with clustering by pretrained unclustering models for daily forecasing at the county level
#
#
#

import sys
# import os
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import random
# from scipy.io import savemat
# from datetime import date
# from keras.models import load_model
# from keras.callbacks import EarlyStopping
# from lib.Time_Period import get_time_periods
# from lib.Network_Uilites import *
# from lib.Models_library import *
from lib.Model_Training import *

# Parameter Setting ====================================================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'                             # GPU Setting

Use_Uni_Pretrained = False                                           # True: use Uni-Pretrained models / False: train Uni-Model
Use_Grained_Pretrained = False                                       # True: use FIGI-Net models / False: train FIGI-Net models

path = 'Data/COVID-19_Dataset_Counties_US.xlsx'                      # Public Data directory
sheet_name = 'Cases'                                                 # Data sheet name
cluster_sheet_name = 'Clusters'                                      # Cluster sheet name
Premodel_save_path = 'Model/Pretrained_Model/'                       # Save uni-model path
Grained_Model_save_path = 'Model/FIGI_Model/'                        # Save FIGI-Net model path
save_path = 'Results/Prediction_Results/'                            # Save prediction outcomes from FIGI-Net model for each time interval
Final_save_path = 'Results/Final_Result/'                            # Save the whole time prediction result

Training_length = 75                                                 # Training data length
predicted_day_fin = 15                                               # Prediction day length
Split_train = 75                                                     # Training data length for data augmentation
window_size = 45                                                     # sliding window for data augmentation
LR = 1e-3                                                            # Learning rate for model training
EPOCHS = 500                                                         # epoch number for model training

# Time train phases ====================================================================================================
time_list = ['041520']
All_Time_Period = ['4/15/20']
Start_day_time = ['2-1-2020']


# time_list = ['041520', '043020', '051520', '053120', '061520', '063020', '071520', '073120', '081520', '083120', '091520', '093020',
#              '101520', '103120', '111520', '113020', '121520', '123120', '011521', '013121', '021521', '022821', '031521', '033121',
#              '041521', '043021', '051521', '053121', '061521', '063021', '071521', '073121', '081521', '083121', '091521', '093021',
#              '101521', '103121', '111521', '113021', '121521', '123121', '011522', '013122', '021522', '022822', '031522', '033122']
# All_Time_Period = ['4/15/20', '4/30/20', '5/15/20', '5/31/20', '6/15/20', '6/30/20', '7/15/20', '7/31/20', '8/15/20', '8/31/20', '9/15/20', '9/30/20',
#                    '10/15/20', '10/31/20', '11/15/20', '11/30/20', '12/15/20', '12/31/20', '1/15/21', '1/31/21', '2/15/21', '2/28/21', '3/15/21', '3/31/21',
#                    '4/15/21', '4/30/21', '5/15/21', '5/31/21', '6/15/21', '6/30/21', '7/15/21', '7/31/21', '8/15/21', '8/31/21', '9/15/21', '9/30/21',
#                    '10/15/21', '10/31/21', '11/15/21', '11/30/21', '12/15/21', '12/31/21', '1/15/22', '1/31/22', '2/15/22', '2/28/22', '3/15/22', '3/31/22']
# Start_day_time = ['2-1-2020', '2-16-2020', '3-2-2020', '3-18-2020', '4-2-2020', '4-17-2020', '5-2-2020', '5-18-2020', '6-2-2020', '6-18-2020', '7-3-2020', '7-18-2020',
#                   '8-2-2020', '8-18-2020', '9-2-2020', '9-17-2020', '10-2-2020', '10-18-2020', '11-2-2020', '11-18-2020', '12-3-2020', '12-16-2020', '12-31-2020', '1-16-2021',
#                   '1-31-2021', '2-15-2021', '3-2-2021', '3-18-2021', '4-2-2021', '4-17-2021', '5-2-2021', '5-18-2021', '6-2-2021', '6-18-2021', '7-3-2021', '7-18-2021',
#                   '8-2-2021', '8-18-2021', '9-2-2021', '9-17-2021', '10-2-2021', '10-18-2021', '11-2-2021', '11-18-2021', '12-3-2021', '12-16-2021', '12-31-2021', '1-16-2022']
start_day = date(2020, 1, 21)

# load the raw data ====================================================================================================

if __name__ == '__main__':

    # Read the data and clean it up
    df_New_cases = pd.read_excel(path, sheet_name=sheet_name, usecols=range(3, 820))
    df_New_cases_diff = df_New_cases.diff(axis=1)
    df_New_cases_diff_denoised = df_New_cases_diff.rolling(7, min_periods=1).mean()
    df_New_cases_diff_denoised = df_New_cases_diff_denoised.to_numpy()
    df_New_cases_diff_denoised = np.nan_to_num(df_New_cases_diff_denoised)
    df_New_cases_diff_denoised_copy = df_New_cases_diff_denoised
    X_time = np.arange(df_New_cases_diff.shape[1], dtype="int64")
    CID_list = np.arange(df_New_cases_diff.shape[0], dtype='int')

    df_New_Clusters = pd.read_excel(path, sheet_name=cluster_sheet_name, usecols=range(2, 50))
    df_New_Clusters = df_New_Clusters.to_numpy()

    # Time Period Index (1-48)
    for c_ind in tqdm(range(len(time_list))):

        Label_idx = df_New_Clusters[:, c_ind]

        Period_time_step_start, Period_time_step_end = get_time_periods(All_Time_Period[c_ind], start_day, Training_length)

        X_time = X_time[Period_time_step_start:Period_time_step_end]
        df_New_cases_diff_denoised_test = df_New_cases_diff_denoised[:, (Period_time_step_end + 1):(Period_time_step_end+predicted_day_fin+1)]
        df_New_cases_diff_denoised_train = df_New_cases_diff_denoised[:, Period_time_step_start:Period_time_step_end]

        df_Predict_Case = np.zeros((df_New_cases_diff_denoised_train.shape[0], predicted_day_fin))
        Prediction_list = []

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path + '{}_Result_Save'.format(time_list[c_ind])):
            os.makedirs(save_path + '{}_Result_Save'.format(time_list[c_ind]))
            print('The new result directory is created !!')


        # Uni_Model_Training ===============================================================================================
        if not Use_Uni_Pretrained:
            Unimodel_save_path = Uni_model_Training(Premodel_save_path, df_New_cases_diff_denoised_train, Bi_sLSTM_Prediction, window_size, predicted_day_fin, LR, EPOCHS, time_list[c_ind])
        else:
            Unimodel_save_path = Premodel_save_path + '{}_Model_Save'.format(time_list[c_ind]) + '/Prediction_UniModel_{}'.format(time_list[c_ind] + '.h5')

        # Grained_Model_Training ===========================================================================================

        for cluster in range(np.max(Label_idx)):

            if not Use_Grained_Pretrained:
                model = Grained_model_Training(Grained_Model_save_path, df_New_cases_diff_denoised_train, Unimodel_save_path, Label_idx, cluster, window_size, predicted_day_fin, LR, EPOCHS, time_list[c_ind])

            else:
                if not os.path.exists(Premodel_save_path):
                    print('Please pretrain the Uni-Model or download the models!!!')
                    sys.exit()
                model = load_model(Grained_Model_save_path + '{}_Model_Save/Prediction_Model_{}_in_Cluster_{}.h5'.format(time_list[c_ind], time_list[c_ind],cluster + 1))

            # print("Forcasting the Prediction for {} !!!".format(time_list[c_ind]))


        # Predict the expected result (Next n days) ========================================================================

            df_New_cases_current_cluster_test_Extra = df_New_cases_diff_denoised_test[np.where(Label_idx[:] == (cluster + 1))]
            df_New_cases_current_cluster_test = df_New_cases_diff_denoised_train[np.where(Label_idx[:] == (cluster + 1))]
            df_New_cases_current_cluster_test = np.hstack((df_New_cases_current_cluster_test, df_New_cases_current_cluster_test_Extra))
            CID_list_cluster = CID_list[np.where(Label_idx[:] == (cluster + 1))]

            df_New_cases_diff_denoised_test_all = np.hstack((df_New_cases_diff_denoised_train, df_New_cases_diff_denoised_test))

            Cluster_List = []
            for county_case_row, cid in zip(df_New_cases_current_cluster_test, CID_list_cluster):
                forecast = []
                for time in range(county_case_row.shape[0] - window_size):
                    temp_predict = model.predict(county_case_row[time:time + window_size][np.newaxis])
                    # record_list = np.zeros((1, 3*predicted_day_fin))
                    record_list = np.zeros((1, (window_size + predicted_day_fin - 1)))
                    if len(forecast) == 0:
                        record_list[:, time:time + predicted_day_fin] = record_list[:, time:time + predicted_day_fin] + temp_predict
                        forecast = record_list
                    else:
                        record_list[:, time:time + predicted_day_fin] = record_list[:, time:time + predicted_day_fin] + temp_predict
                        forecast = np.vstack((forecast, record_list))

                forecast_new = forecast[:window_size-predicted_day_fin-1, window_size-2*predicted_day_fin-1:window_size-predicted_day_fin-1]

                Cluster_results = []
                All_Cluster_sliding = []
                for all_sliding in range(forecast_new.shape[1]):
                    Cluster_results_tmp = []
                    for one_sliding_day in range(forecast_new.shape[1]):
                        Cluster_results_tmp.append(forecast_new[((forecast_new.shape[1]-1-all_sliding)+one_sliding_day), one_sliding_day])

                    if len(All_Cluster_sliding) == 0:
                        All_Cluster_sliding = Cluster_results_tmp
                    else:
                        All_Cluster_sliding = np.vstack((All_Cluster_sliding, Cluster_results_tmp))

                Cluster_results = All_Cluster_sliding[-1, :]

        # Save the outcomes ================================================================================================
                if len(Cluster_List) == 0:
                    Cluster_List = Cluster_results
                else:
                    Cluster_List = np.vstack((Cluster_List, Cluster_results))

                savemat(save_path + '{}_Result_Save/County_{}_in_Cluster_{}.mat'.format(time_list[c_ind], cid, cluster), mdict={'Sliding_predicted_list': All_Cluster_sliding})

            print("Cluster {} in Time Period {} is completed!!! \n".format(cluster, time_list[c_ind]))

            df_Predict_Case[np.where(Label_idx[:] == (cluster + 1))] = Cluster_List

            if cluster == 0:
                if Cluster_List.ndim != 1:
                    Prediction_list = np.mean(Cluster_List, axis=0)
                else:
                    Prediction_list = Cluster_List
            else:
                if Cluster_List.ndim != 1:
                    Prediction_list = np.vstack((Prediction_list, np.mean(Cluster_List, axis=0)))
                else:
                    Prediction_list = np.vstack((Prediction_list, Cluster_List))

        savemat(save_path + 'Prediction_w_slidingday_{}.mat'.format(time_list[c_ind]), mdict={'Prediction_list': Prediction_list, 'df_New_cases_diff_denoised_copy': df_New_cases_diff_denoised_copy,
                                                                                              'df_Predict_Case': df_Predict_Case, 'Label_idx': Label_idx})

        sleep(0.001)

    # Combine time intervals to generate whole prediction results for all time length ==================================

    Post_processing(Final_save_path, save_path, time_list, All_Time_Period, df_New_cases_diff_denoised)

