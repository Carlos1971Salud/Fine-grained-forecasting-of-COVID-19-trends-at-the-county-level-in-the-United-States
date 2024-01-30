
import tensorflow as tf

def Bi_sLSTM_Prediction(Node_number, predict_day):

    model = tf.keras.models.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(Node_number, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(Node_number)),
            tf.keras.layers.Dense(predict_day),
            tf.keras.layers.Lambda(lambda x: x * 100.0)
            ])

    return model

def GRU_Model(Node_number, predict_day):

    model = tf.keras.models.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
            tf.keras.layers.GRU(Node_number),
            tf.keras.layers.Dense(predict_day),
            tf.keras.layers.Lambda(lambda x: x * 100.0)
            ])

    return model

def LSTM_Model(Node_number, predict_day):

    model = tf.keras.models.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
            tf.keras.layers.LSTM(Node_number),
            tf.keras.layers.Dense(predict_day),
            tf.keras.layers.Lambda(lambda x: x * 100.0)
            ])

    return model