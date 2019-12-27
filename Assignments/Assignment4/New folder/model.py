import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))


        ### TODO(Students) START
        self.bidirectional = layers.Bidirectional(layers.GRU(hidden_size, return_sequences=True))
        self.tanh = tf.keras.activations.tanh
        # ...
        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        # ...
        H = rnn_outputs
        M = self.tanh(H)
        alpha = tf.matmul(M, self.omegas)
        softmax_alpha = tf.nn.softmax(alpha)
        r = tf.math.multiply(H, softmax_alpha)
        h_star = self.tanh(r)
        output = tf.math.reduce_sum(h_star,axis=1)

        ### TODO(Students) END

        return output

    def call(self, inputs, pos_inputs, training):
        # import pdb; pdb.set_trace()
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START
        # ...
        embed = tf.concat([word_embed, pos_embed], axis=2)
        mask = tf.cast(inputs!=0, dtype=tf.dtypes.bool)
        embed=tf.cast(embed, dtype=tf.dtypes.float64)
        rnn_outputs = self.bidirectional(embed,training=training, mask=mask)
        attn_op = self.attn(rnn_outputs)
        logits = self.decoder(attn_op)
        ### TODO(Students) END

        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START
        # ...
        self.num_classes = len(ID_TO_CLASS)
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))
        self.dropout = layers.Dropout(0.4)
        self.omegas = tf.Variable(tf.random.normal((100, 1)))
        # self.conv1 = layers.Conv2D(input_shape=(None, 200, 1), filters=1, kernel_size=(2, 128))
        # self.conv2 = layers.Conv2D(input_shape=(None, 200, 1), filters=1, kernel_size=(3, 128))
        # self.conv3 = layers.Conv2D(input_shape=(None, 200, 1), filters=1, kernel_size=(4, 128))
        # self.conv4 = layers.Conv2D(input_shape=(None, 200, 1), filters=1, kernel_size=(2, 128))
        # self.conv5 = layers.Conv2D(input_shape=(None, 200, 1), filters=1, kernel_size=(3, 128))
        # self.conv6 = layers.Conv2D(input_shape=(None, 200, 1), filters=1, kernel_size=(4, 128))
        #
        # self.conv12 = layers.Conv2D(input_shape=(None, 100, 1), filters=1, kernel_size=(2, 32))
        # self.conv22 = layers.Conv2D(input_shape=(None, 100, 1), filters=1, kernel_size=(3, 32))
        # self.conv32 = layers.Conv2D(input_shape=(None, 100, 1), filters=1, kernel_size=(4, 32))
        # self.conv42 = layers.Conv2D(input_shape=(None, 100, 1), filters=1, kernel_size=(2, 32))
        # self.conv52 = layers.Conv2D(input_shape=(None, 100, 1), filters=1, kernel_size=(3, 32))
        # self.conv62 = layers.Conv2D(input_shape=(None, 100, 1), filters=1, kernel_size=(4, 32))

        self.conv12 = layers.Conv2D(input_shape=(None, 200, 1), filters=1, kernel_size=(2, 32))
        self.conv22 = layers.Conv2D(input_shape=(None, 200, 1), filters=1, kernel_size=(3, 32))
        self.conv32 = layers.Conv2D(input_shape=(None, 200, 1), filters=1, kernel_size=(4, 32))
        self.conv42 = layers.Conv2D(input_shape=(None, 200, 1), filters=1, kernel_size=(2, 32))
        self.conv52 = layers.Conv2D(input_shape=(None, 200, 1), filters=1, kernel_size=(3, 32))
        self.conv62 = layers.Conv2D(input_shape=(None, 200, 1), filters=1, kernel_size=(4, 32))

        self.maxpool1 = layers.GlobalMaxPool1D()

        self.flatten = layers.Flatten()
        self.tanh = tf.keras.activations.tanh
        self.relu = tf.keras.activations.relu
        self.relu = tf.keras.layers.LeakyReLU()
        self.linear0 = layers.Dense(200)
        self.linear1 = layers.Dense(100)
        self.linear2 = layers.Dense(50)
        self.linear = layers.Dense(self.num_classes)

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        # ...
        rnn_outputs = rnn_outputs[:, :, :, 0]
        H = rnn_outputs

        M = self.tanh(H)
        alpha = tf.matmul(M, self.omegas)
        softmax_alpha = tf.nn.softmax(alpha)
        r = tf.math.multiply(H, softmax_alpha)
        h_star = self.tanh(r)
        output = tf.math.reduce_sum(h_star, axis=1)

        ### TODO(Students) END

        return output


        ### TODO(Students END

    def call(self, inputs, pos_inputs, training):
        # raise NotImplementedError
        ### TODO(Students) START
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)
        embed = tf.concat([word_embed, pos_embed], axis=2)
        mask = tf.cast(inputs != 0, dtype=tf.dtypes.bool)
        embed = tf.cast(embed, dtype=tf.dtypes.double)
        # o= self.dropout(embed)
        embed = tf.expand_dims(embed,3)
        word_embed = tf.expand_dims(word_embed,3)
        pos_embed = tf.expand_dims(pos_embed, 3)


        c1 = self.conv12(embed)
        t1 = self.relu(c1)
        t2 = self.conv22(embed)
        m1 = self.maxpool1(t1[:, :, :, 0])
        t2= self.relu(t2)
        m2 = self.maxpool1(t2[:, :, :, 0])
        c3 = self.conv32(embed)
        t3 = self.relu(c3)
        m3 = self.maxpool1(t3[:, :, :, 0])
        c4 = self.conv42(embed)
        t4 = self.relu(c4)
        m4 = self.maxpool1(t4[:, :, :, 0])
        c5 = self.conv52(embed)
        t5 = self.relu(c5)
        m5 = self.maxpool1(t5[:, :, :, 0])
        c6 = self.conv62(embed)
        t6 = self.relu(c6)
        m6 = self.maxpool1(t6[:, :, :, 0])

        temp=tf.concat([m1,m2,m3, m4,m5,m6], axis=1)
        temp = self.tanh(temp)
        att = self.attn(word_embed)
        att_p = self.attn(pos_embed)
        temp = tf.concat([temp, att, att_p], axis=1)
        fl = self.flatten(temp)
        fl = self.dropout(fl)
        # l1 = self.linear1(fl)
        f = self.linear(fl)
        return {'logits': f}

        ### TODO(Students END
