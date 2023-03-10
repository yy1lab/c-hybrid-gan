from relational_memory import *
from utils import *

"""## Generator"""


class Generator(tf.keras.models.Model):
    def __init__(self, emb_units, proj_units, emb_dropout_rate, proj_dropout_rate,
                 mem_slots, head_size, num_heads, num_blocks, num_tokens, num_meta_features):
        super(Generator, self).__init__()

        self.p_subg = SubGenerator(emb_units[0], proj_units[0], emb_dropout_rate[0],
                                   proj_dropout_rate[0], mem_slots[0], head_size[0],
                                   num_heads[0], num_blocks[0], num_tokens[0],
                                   num_meta_features)

        self.d_subg = SubGenerator(emb_units[1], proj_units[1], emb_dropout_rate[1],
                                   proj_dropout_rate[1], mem_slots[1], head_size[1],
                                   num_heads[1], num_blocks[1], num_tokens[1],
                                   num_meta_features)

        self.r_subg = SubGenerator(emb_units[2], proj_units[2], emb_dropout_rate[2],
                                   proj_dropout_rate[2], mem_slots[2], head_size[2],
                                   num_heads[2], num_blocks[2], num_tokens[2],
                                   num_meta_features)

    def call(self, inputs, memory, training=False):
        """
        :param inputs: meta and note data
        :type inputs : tuple
        :shape inputs: ([None, NUM_META_FEATURES], 
                        ([None, NUM_P_TOKENS], [None, NUM_D_TOKENS], [None, NUM_R_TOKENS]))

        :param memory: memory for rmc core for each sub generator
        :type  memory: tuple
        """

        m, (p, d, r) = inputs
        memory_p, memory_d, memory_r = memory

        p, memory_p = self.p_subg((m, p), memory_p, training=training)
        d, memory_d = self.d_subg((m, d), memory_d, training=training)
        r, memory_r = self.r_subg((m, r), memory_r, training=training)

        return (p, d, r), (memory_p, memory_d, memory_r)


"""### SubGenerator"""


class SubGenerator(tf.keras.models.Model):
    def __init__(self, emb_units, proj_units, emb_dropout_rate, proj_dropout_rate,
                 mem_slots, head_size, num_heads, num_blocks, num_tokens, num_meta_features):
        super(SubGenerator, self).__init__()

        self.embedding = tf.keras.layers.Dense(
            emb_units, use_bias=False, kernel_initializer=create_linear_initializer(num_tokens))

        self.embedding_dropout = tf.keras.layers.Dropout(emb_dropout_rate)

        self.projection = tf.keras.layers.Dense(
            proj_units, activation='relu', kernel_initializer=create_linear_initializer(emb_units + num_meta_features))

        self.projection_dropout = tf.keras.layers.Dropout(proj_dropout_rate)

        self.rmc = RelationalMemory(mem_slots, head_size, num_heads, num_blocks)

        self.outputs = tf.keras.layers.Dense(
            num_tokens, kernel_initializer=create_linear_initializer(mem_slots * head_size * num_heads))

    def call(self, inputs, memory, training=False):
        """
        :param inputs: meta (i.e. syllable) and note attribute (pitch, duration or rest)
        :type. inputs: tuple
        :shape inputs: ([None, NUM_META_FEATURES], [None, NUM_[.]_TOKENS])

        :param memory: rmc memory
        """
        m, n = inputs

        n = self.embedding(n)
        n = self.embedding_dropout(n, training=training)

        x = tf.concat([n, m], axis=-1)  # [p+d+r+m]

        x = self.projection(x)
        x = self.projection_dropout(x, training=training)

        x, memory = self.rmc(x, memory)

        x = self.outputs(x)

        return x, memory
