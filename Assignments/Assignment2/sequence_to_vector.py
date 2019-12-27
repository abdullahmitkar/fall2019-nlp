# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models


class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):

    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        # ...
        self.num_layer = num_layers;
        self.dropout = dropout;
        self._input_dim = input_dim;
        self.layer = tf.keras.layers.Dense(input_dim, input_dim=input_dim, activation='relu')
        self.soft_max_layer = tf.keras.layers.Dense(input_dim,input_dim=input_dim, activation = 'softmax')

        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start

        batch_size = vector_sequence.shape[0]
        dropout_mask = []
        for x in np.random.uniform(1, 0, vector_sequence.shape[1]):
            if x > self.dropout:
                dropout_mask.append(1)
            else:
                dropout_mask.append(0)
        dropout_mask_batch_size = []
        for _ in range(batch_size):
            dropout_mask_batch_size.append(dropout_mask)
        vec_dropout_mask = tf.convert_to_tensor(tf.dtypes.cast(dropout_mask_batch_size, tf.dtypes.int32));
        # print("Vec", vec_dropout_mask.shape)
        training = False
        if training:
            mask_ = vec_dropout_mask * tf.dtypes.cast(sequence_mask, tf.dtypes.int32)
        else:
            mask_ = sequence_mask;
        # print("Mask", mask_.shape)
        vector_sequence_masked = tf.ragged.boolean_mask(vector_sequence, tf.dtypes.cast(mask_,tf.dtypes.bool))
        vector_sequence_masked = vector_sequence;
        # print("VS Shape", vector_sequence_masked.shape)
        av_list = []
        for i in range(batch_size):
            av = tf.reduce_mean(vector_sequence_masked[i], 0)
            av_list.append(av)
        # print(len(av_list)) ## 64

        n_layer = av_list
        n_layer= tf.stack(n_layer) #64, 128
        # print(n_layer.shape)
        layer_representations = []
        for _ in range(self.num_layer):

            n_layer = self.layer(n_layer)
            # print(n_layer.shape , "Stacking")
            layer_representations.append(n_layer)

        combined_vector = n_layer;
        # print("Combined Vector shape",combined_vector.shape)
        layer_representations = tf.stack(layer_representations);

        layer_representations = tf.transpose(layer_representations,perm=[1,0,2]);
        # print(layer_representations.shape)
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        # ...
        self.num_layer = num_layers;
        self._input_dim = input_dim;
        self.tanh = tf.keras.layers.GRU(input_dim,  input_dim=input_dim,
                                        return_sequences=True, return_state=True ,activation='tanh')

        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        # ...
        batch_size = vector_sequence.shape[0]

        layer_representations = []
        n_layer=vector_sequence;
        for i in range(self.num_layer):
            n_plus_one_layer = self.tanh(n_layer,mask=sequence_mask,training=training)
            # print(n_plus_one_layer)
            n_layer = n_plus_one_layer
            layer_representations.append(n_plus_one_layer[1])
        combined_vector = n_layer[1];
        layer_representations = tf.stack(layer_representations);

        layer_representations = tf.transpose(layer_representations, perm=[1, 0, 2]);
        # print(layer_representations.shape)
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
