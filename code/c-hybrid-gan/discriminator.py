from relational_memory import *
from utils import *

"""## Discriminator"""


class Discriminator(tf.keras.models.Model):
    def __init__(self, emb_units, proj_units, emb_dropout_rate, proj_dropout_rate,
                 mem_slots, head_size, num_heads, num_blocks, num_tokens, num_meta_features):
        super(Discriminator, self).__init__()

        self.embedding = [tf.keras.layers.Dense(
            units, use_bias=False, kernel_initializer=create_linear_initializer(n)) for units, n in
            zip(emb_units, num_tokens)]

        self.embedding_dropout = [tf.keras.layers.Dropout(rate) for rate in emb_dropout_rate]

        self.projection = tf.keras.layers.Dense(
            proj_units, activation='relu',
            kernel_initializer=create_linear_initializer(sum(emb_units) + num_meta_features))

        self.projection_dropout = tf.keras.layers.Dropout(proj_dropout_rate)

        self.rmc = RelationalMemory(mem_slots, head_size, num_heads, num_blocks)

        self.outputs = tf.keras.layers.Dense(1)

    def call(self, inputs, memory, training=False):
        """
        :param inputs: meta and note data
        :type inputs : tuple
        :shape inputs: ([None, NUM_META_FEATURES], 
                        ([None, NUM_P_TOKENS], [None, NUM_D_TOKENS], [None, NUM_R_TOKENS]))

        :param memory: memory for rmc layer
        """
        m, (p, d, r) = inputs

        p = self.embedding[0](p)
        d = self.embedding[1](d)
        r = self.embedding[2](r)

        p = self.embedding_dropout[0](p, training=training)
        d = self.embedding_dropout[1](d, training=training)
        r = self.embedding_dropout[2](r, training=training)

        x = tf.concat([p, d, r, m], axis=-1)  # [pitch, duration, rest ; meta]

        x = self.projection(x)
        x = self.projection_dropout(x, training=training)

        x, memory = self.rmc(x, memory)

        x = self.outputs(x)

        return x, memory
