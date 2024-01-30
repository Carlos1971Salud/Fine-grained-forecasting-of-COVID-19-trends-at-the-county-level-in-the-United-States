
import random
from keras.models import load_model
from keras.callbacks import EarlyStopping
from lib.Time_Period import get_time_periods
from lib.Network_Uilites import *
from lib.Models_library import *


def Uni_model_Training(save_path, df_New_cases_diff_denoised_train, Bi_sLSTM_Prediction, window_size, predicted_day_fin, LR, EPOCHS, time_period_label):

    All_dataset_Xtrain, All_dataset_Ytrain = [], []
    for county_case_row in df_New_cases_diff_denoised_train:
        County_x, County_y = train_window_ungraphing(county_case_row, ref_day=window_size, predict_day=predicted_day_fin)
        if len(All_dataset_Xtrain) == 0:
            All_dataset_Xtrain = County_x
            All_dataset_Ytrain = County_y
        else:
            All_dataset_Xtrain = np.concatenate((All_dataset_Xtrain, County_x))
            All_dataset_Ytrain = np.vstack((All_dataset_Ytrain, County_y))

    # Randomly reorder the training data
    ind = list(range(All_dataset_Ytrain.shape[0]))
    random.shuffle(ind)
    X_dataset_train = All_dataset_Xtrain[ind, :]
    Y_dataset_train = All_dataset_Ytrain[ind, :]

    X_dataset_train, Y_dataset_train = shuffle(All_dataset_Xtrain, All_dataset_Ytrain, shuffle_buffer=1000)

    # Model Setting and Training =======================================================================================

    model = Bi_sLSTM_Prediction(window_size, predicted_day_fin)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    model.compile(loss="mae", optimizer=optimizer, metrics=['mse'])
    history = model.fit(X_dataset_train,
                        Y_dataset_train,
                        epochs=EPOCHS,
                        batch_size=128,
                        validation_split=0.2,
                        callbacks=[EarlyStopping(patience=100, restore_best_weights=True)],
                        verbose=1)

    loss = history.history["loss"]
    epochs = range(10, len(loss))
    plot_loss = loss[10:]

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path + '{}_Model_Save'.format(time_period_label)):
        os.makedirs(save_path + '{}_Model_Save'.format(time_period_label))
        print('The new directory is created !!')
    # if not os.path.exists(save_path + '{}_Model_Save/Results_unimodel'.format(time_period_label)):
    #     os.makedirs(save_path + '{}_Model_Save/Results_unimodel'.format(time_period_label))
    #     print('The new directory is created !!')

    model_save_path = save_path + '{}_Model_Save/Prediction_UniModel_{}.h5'.format(time_period_label, time_period_label)
    model.save(model_save_path)
    print("Pretrained Uni-Model is saved in {}!!!".format(time_period_label))

    return model_save_path

def Grained_model_Training(save_path, df_New_cases_diff_denoised_train, Unimodel_save_path, Label_idx, cluster, window_size, predicted_day_fin, LR, EPOCHS, time_period_label):

    All_dataset_Xtrain, All_dataset_Ytrain = [], []
    df_New_cases_current_cluster = df_New_cases_diff_denoised_train[np.where(Label_idx[:] == (cluster + 1))]
    for county_case_row in df_New_cases_current_cluster:
        County_x, County_y = train_window_ungraphing(county_case_row, ref_day=window_size,
                                                     predict_day=predicted_day_fin)
        if len(All_dataset_Xtrain) == 0:
            All_dataset_Xtrain = County_x
            All_dataset_Ytrain = County_y
        else:
            All_dataset_Xtrain = np.vstack((All_dataset_Xtrain, County_x))
            All_dataset_Ytrain = np.vstack((All_dataset_Ytrain, County_y))

    # Randomly reorder the training data
    ind = list(range(All_dataset_Ytrain.shape[0]))
    random.shuffle(ind)
    X_dataset_train = All_dataset_Xtrain[ind, :]
    Y_dataset_train = All_dataset_Ytrain[ind, :]

    X_dataset_train, Y_dataset_train = shuffle(All_dataset_Xtrain, All_dataset_Ytrain, shuffle_buffer=1000)

    # Pretrained Model Setting =========================================================================================

    model = load_model(Unimodel_save_path)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    model.compile(loss="mae", optimizer=optimizer, metrics=['mse'])
    history = model.fit(X_dataset_train,
                        Y_dataset_train,
                        epochs=EPOCHS,
                        batch_size=128,
                        validation_split=0.2,
                        callbacks=[EarlyStopping(patience=100, restore_best_weights=True)],
                        verbose=1)

    loss = history.history["loss"]
    epochs = range(10, len(loss))
    plot_loss = loss[10:]

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path + '{}_Model_Save'.format(time_period_label)):
        os.makedirs(save_path + '{}_Model_Save'.format(time_period_label))
        print('The new directory is created !!')
    # if not os.path.exists(save_path + '{}_Model_Save/Results'.format(time_period_label)):
    #     os.makedirs(save_path + '{}_Model_Save/Results'.format(time_period_label))


    model.save \
        (save_path + '{}_Model_Save/Prediction_Model_{}_in_Cluster_{}.h5'.format(time_period_label, time_period_label, cluster + 1))
    print("Forcasting the Prediction for Cluster {} in {} !!!".format((cluster + 1), time_period_label))

    return model
