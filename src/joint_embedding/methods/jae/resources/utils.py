"""
Utilities for consistent data preprocessing
"""

import numpy as np
import tensorflow as tf

class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


class JointEmbeddingModel(tf.keras.Model):
    def __init__(self, params, name=None):
        super(JointEmbeddingModel, self).__init__(name=name)
        self.params = params
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
        self.classifier = self.create_classifier()

    def get_config(self):
        return {
                "params": self.params,
        }
    def call(self, inputs, training):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        digits_cell_type, digits_batch, digits_phase = self.classifier(encoded)
        if self.params['use_batch']:
            return decoded, digits_cell_type, digits_batch, digits_phase
        else:
            return decoded, digits_cell_type

    def create_encoder(self, use_resnet=True):
        if use_resnet:
            inputs = tf.keras.layers.Input(shape=(self.params['dim'],))
            for i, n_unit in enumerate(self.params['hidden_units'][:-1]):
                if i==0:
                    x_init = tf.keras.layers.Dense(n_unit, activation='relu')(inputs)
                else:
                    x_init = tf.keras.layers.Dense(n_unit, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.1)(x_init)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dense(n_unit)(x)
                x = tf.keras.layers.Add()([x,x_init])
                x = tf.keras.layers.Activation(activation='relu')(x)
            encoded = tf.keras.layers.Dense(self.params['hidden_units'][-1], activation='relu')(x)
        else:
            inputs = tf.keras.layers.Input(shape=(self.params['dim'],))
            for i, n_unit in enumerate(self.params['hidden_units'][:-1]):
                if i==0:
                    x = tf.keras.layers.Dense(n_unit, activation='relu')(inputs)
                else:
                    x = tf.keras.layers.Dense(n_unit, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.1)(x)
                x = tf.keras.layers.BatchNormalization()(x)
            encoded = tf.keras.layers.Dense(self.params['hidden_units'][-1], activation='relu')(x)
        return tf.keras.Model(inputs=inputs, outputs=encoded, name='encoder')

    def create_decoder(self):
        inputs = tf.keras.layers.Input(shape=(self.params['hidden_units'][-1],))
        for i, n_unit in enumerate(self.params['hidden_units'][:-1][::-1]):
            if i==0:
                x = tf.keras.layers.Dense(n_unit, activation='relu')(inputs)
            else:
                x = tf.keras.layers.Dense(n_unit, activation='relu')(x)
        decoded = tf.keras.layers.Dense(self.params['dim'], activation='relu')(x)
        return tf.keras.Model(inputs=inputs, outputs=decoded, name='decoder')

    def create_classifier(self):
        inputs = tf.keras.layers.Input(shape=(self.params['hidden_units'][-1],))
        digits_cell_type = inputs[:,:self.params['nb_cell_types']]
        digits_batch = inputs[:,self.params['nb_cell_types']:(self.params['nb_cell_types']+self.params['nb_batches'])]
        digits_phase = inputs[:,(self.params['nb_cell_types']+self.params['nb_batches']):(self.params['nb_cell_types']+self.params['nb_batches']+self.params['nb_phases'])]
        return tf.keras.Model(inputs=inputs, outputs=[digits_cell_type, digits_batch, digits_phase], name='classifier')

