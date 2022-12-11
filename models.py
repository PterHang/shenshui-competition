import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, losses
from config import time_steps, pred_time_steps, n_features


class Encoder(Model):
    def __init__(self):
        super().__init__()
        self.lstm = layers.LSTM(units=64, input_shape=(time_steps, n_features), return_sequences=True,
                                return_state=True, kernel_initializer='he_normal')

    def call(self, input_seq):
        print(self.lstm(input_seq))
        encode_outputs, h, c = self.lstm(input_seq)

        return encode_outputs, h, c


class Decoder(Model):
    def __init__(self):
        super().__init__()
        self.lstm = layers.LSTM(units=64, input_shape=(time_steps, n_features), return_sequences=True,
                                return_state=True, kernel_initializer='he_normal')
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(pred_time_steps)
        # self.attention = layers.Attention()

    def call(self, input_seq, h, c):
        batch_size = tf.shape(input_seq)[0]
        input_seq = tf.reshape(input_seq, [batch_size, 1, n_features])
        decode_ouputs, h, c = self.lstm(input_seq, initial_state=[h, c])
        pred = self.fc1(decode_ouputs)  # pred(batch_size, 1, output_size)
        pred = self.fc2(pred)
        pred = pred[:, -1, :]

        return pred, h, c


class Seq2Seq(Model):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, input_seq):
        seq_len = input_seq.shape[1]
        encode_ouputs, h, c = self.encoder(input_seq)
        res = None
        for t in range(seq_len):
            _input = input_seq[:, t, :]
            decode_outputs, h, c = self.decoder(_input, h, c)
            res = decode_outputs
        return res


class BiLSTM(Model):
    def __init__(self):
        super().__init__()
        self.bilstm = layers.Bidirectional(
            layers.LSTM(128, input_shape=(time_steps, n_features), return_sequences=True))
        # self.bilstm = layers.LSTM(128, input_shape=(time_steps, n_features), return_sequences=True, kernel_initializer='he_uniform')
        # self.bilstm = layers.LSTM(128, return_sequences=True, kernel_initializer='he_uniform')
        self.fc0 = layers.Dense(64, activation='relu')
        self.fc1 = layers.Dense(pred_time_steps)

    def call(self, inputs):
        x = self.bilstm(inputs)
        x = self.fc0(x)
        outputs = self.fc1(x)
        return outputs[:, -1, :]


def lstm_model():
    model = Sequential()
    model.add(layers.LSTM(200, input_shape=(time_steps, n_features)))
    model.add(layers.RepeatVector(pred_time_steps))
    model.add(layers.LSTM(200, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(100, activation='relu')))
    model.add(layers.Dropout(0.5))
    model.add(layers.TimeDistributed(layers.Dense(1)))
    return model
