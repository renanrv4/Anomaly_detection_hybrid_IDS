import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

class GRUModel(tf.keras.Model):

    def __init__(self, num_ids, embedding_dim, gru_units, dropout):
        super().__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=num_ids,
            output_dim=embedding_dim
        )

        self.gru = tf.keras.layers.GRU(
            gru_units,
            return_sequences=False
        )

        self.dropout = tf.keras.layers.Dropout(
            dropout
        )

        self.output_layer = tf.keras.layers.Dense(
            num_ids,
            activation="softmax"
        )
    
    def call(self, inputs, training=False):
        # Create embedding
        x = self.embedding(inputs)

        # Send input to GRU layer
        x = self.gru(x)

        # Dropout layer
        x = self.dropout(x, training=training)

        # Output of Dense Layer
        return self.output_layer(x)
        