# Collect the data to generate the feature vectors for temporal clustering
#
#
#

from lib.Network_Uilites import *

def data_collection(data_path, save_path, time_list, window_size):

    # load the raw data ====================================================================================================
    # Read the data and clean it up
    df_New_cases = pd.read_excel(data_path, sheet_name='Cases', usecols=range(3, 820))
    df_New_cases_all = df_New_cases.to_numpy()
    df_New_cases_diff = df_New_cases.diff(axis=1)
    df_New_cases_diff_1 = df_New_cases_diff.to_numpy()
    df_New_cases_diff_1 = df_New_cases_diff_1[:, 1:]
    df_New_cases_diff_denoised = df_New_cases_diff.rolling(window_size, min_periods=1).mean()
    df_New_cases_diff_denoised = df_New_cases_diff_denoised.to_numpy()
    df_New_cases_diff_denoised = np.nan_to_num(df_New_cases_diff_denoised)
    df_New_cases_diff_act = df_New_cases_diff.to_numpy()

    df_New_death = pd.read_excel(data_path, sheet_name='Death', usecols=range(3, 820))
    df_New_death_all = df_New_death.to_numpy()
    df_New_death_diff = df_New_death.diff(axis=1)
    df_New_death_diff_1 = df_New_death_diff.to_numpy()
    df_New_death_diff_1 = df_New_death_diff_1[:, 1:]

    df_New_CID = pd.read_excel(data_path, sheet_name='CID', usecols=range(2, 3))
    df_New_CID = df_New_CID.to_numpy()
    df_New_Pop = pd.read_excel(data_path, sheet_name='Population', usecols=range(3, 4))
    df_New_Pop = df_New_Pop.to_numpy()
    df_New_area = pd.read_excel(data_path, sheet_name='Area', usecols=range(3, 4))
    df_New_area = df_New_area.to_numpy()
    df_New_Popdense = df_New_Pop/df_New_area
    select_index = pd.read_excel(data_path, sheet_name='Valid State')


    df_New_cases_diff_1_denoised = rolling_value(df_New_cases_diff_1, window_size)
    tmp_last_case_denoised = df_New_cases_diff_1_denoised[:, -1]
    df_New_cases_diff_1_denoised = np.hstack((df_New_cases_diff_1_denoised[:, 1:], tmp_last_case_denoised[:, np.newaxis]))
    df_New_death_diff_1_denoised = rolling_value(df_New_death_diff_1, window_size)
    tmp_last_death_denoised = df_New_death_diff_1_denoised[:, -1]
    df_New_death_diff_1_denoised = np.hstack((df_New_death_diff_1_denoised[:, 1:], tmp_last_death_denoised[:, np.newaxis]))

    df_New_fatal_all = df_New_death_all/df_New_cases_all
    df_New_fatal_all[np.isnan(df_New_fatal_all)] = 0
    df_New_fatal_all = df_New_fatal_all * 100

    for i in tqdm(range(len(time_list))):

        df_New_cases_seg = df_New_cases_all[:, 10+15*i:10+75+1+15*i]
        df_New_death_seg = df_New_death_all[:, 10+15*i:10+75+1+15*i]
        df_New_fatal_seg = df_New_fatal_all[:, 10+15*i:10+75+1+15*i]
        nop_increase_total_trimmed = df_New_cases_diff_1_denoised[:, 11+15*i:12+75+15*i]
        df_New_death_diff_1_denoised_seg = df_New_death_diff_1_denoised[:, 11+15*i:12+75+15*i]
        nop_increase_total_trimmed_pre = np.zeros((df_New_cases_seg.shape[0], df_New_cases_seg.shape[1]))

        savemat(save_path+'note_{}_01_C_NewData.mat'.format(time_list[i]), mdict={'CID': df_New_CID,
                                                                                      'Pop': df_New_Pop,
                                                                                      'Popdens': df_New_Popdense,
                                                                                      'cases': df_New_cases_seg,
                                                                                      'cases_all': df_New_cases_all,
                                                                                      'cases_denoise': df_New_cases_diff_1_denoised,
                                                                                      'cases_increase': df_New_cases_diff_1,
                                                                                      'selected_idx': select_index,
                                                                                      'nop_increase_total_trimmed': nop_increase_total_trimmed,
                                                                                      'nop_increase_total_trimmed_pre': nop_increase_total_trimmed_pre,
                                                                                      'fatal': df_New_fatal_seg,
                                                                                      'fatal_all': df_New_fatal_all,
                                                                                      'death_increase': df_New_death_diff_1,
                                                                                      'death_denoise': df_New_death_diff_1_denoised,
                                                                                      'death_denoise_period': df_New_death_diff_1_denoised_seg,
                                                                                      'death_all': df_New_death_all,
                                                                                      'death': df_New_death_seg})


        # Feature vector geneneration ======================================================================================

        acf_feature_dynamic = []
        ccf_feature_dynamic = []
        acf_feature_dynamic_fatal = []

        for sample_ind in range(nop_increase_total_trimmed.shape[0]):

                tmp_nop_increase_total_trimmed = nop_increase_total_trimmed[sample_ind, :] #/ np.max(nop_increase_total_trimmed[sample_ind, :])
                tmp_df_New_fatal_seg = df_New_fatal_seg[sample_ind, :] #/ np.max(df_New_fatal_seg[sample_ind, :])

                tmp_acf_feature_dynamic = autocorrelation_function(tmp_nop_increase_total_trimmed)
                tmp_acf_feature_dynamic_fatal = autocorrelation_function(tmp_df_New_fatal_seg)
                tmp_ccf_feature_dynamic = correlate(tmp_nop_increase_total_trimmed, tmp_df_New_fatal_seg, mode='full', method='fft')


                tmp_acf_feature_dynamic[np.isnan(tmp_acf_feature_dynamic)] = 0
                tmp_ccf_feature_dynamic[np.isnan(tmp_ccf_feature_dynamic)] = 0
                tmp_acf_feature_dynamic_fatal[np.isnan(tmp_acf_feature_dynamic_fatal)] = 0

                if len(acf_feature_dynamic)==0:
                    acf_feature_dynamic = tmp_acf_feature_dynamic
                    ccf_feature_dynamic = tmp_ccf_feature_dynamic
                    acf_feature_dynamic_fatal = tmp_acf_feature_dynamic_fatal
                else:
                    acf_feature_dynamic = np.vstack((acf_feature_dynamic, tmp_acf_feature_dynamic))
                    ccf_feature_dynamic = np.vstack((ccf_feature_dynamic, tmp_ccf_feature_dynamic))
                    acf_feature_dynamic_fatal = np.vstack((acf_feature_dynamic_fatal, tmp_acf_feature_dynamic_fatal))

        all_acf_feature_dynamic = np.hstack((acf_feature_dynamic[:, 1:], acf_feature_dynamic_fatal[:, 1:]))
        feature_dynamic_fatal = np.hstack((acf_feature_dynamic_fatal, ccf_feature_dynamic))
        feature_dynamic = np.hstack((all_acf_feature_dynamic, ccf_feature_dynamic))

        savemat(save_path+'note_{}_02_C_New_features.mat'.format(time_list[i]), mdict={'CID': df_New_CID,
                                                                                       'acf_feature_dynamic': all_acf_feature_dynamic,
                                                                                       'ccf_feature_dynamic': ccf_feature_dynamic,
                                                                                       'feature_dynamic': feature_dynamic,
                                                                                       'feature_dynamic_fatal': feature_dynamic_fatal,
                                                                                       'selected_idx': select_index})
        time.sleep(0.001)

    print('Data and Feature Collections are finished ======================================================================')