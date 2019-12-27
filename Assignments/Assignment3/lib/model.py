# inbuilt lib imports:
from typing import Dict
import math

# external libs
import tensorflow as tf
from tensorflow.keras import models, layers

# project imports


class CubicActivation(layers.Layer):
    """
    Cubic activation as described in the paper.
    """
    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        vector : ``tf.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start
        # Comment the next line after implementing call.
        cubing_tensor = tf.constant([3.0])
        cube = tf.pow(vector, cubing_tensor)
        return cube;
        # raise NotImplementedError
        # TODO(Students) End



class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students) Start
        # self.weights0 = tf.Variable(tf.random_normal([2, 3], stddev=0.1),name="weights0")


        self.weights1 = tf.Variable(tf.random.truncated_normal([num_tokens* embedding_dim,hidden_dim ],mean = 0,stddev=1.0 / math.sqrt(hidden_dim)),trainable = True)
        # self.weights1 = tf.Variable(tf.random.uniform(shape=(num_tokens* embedding_dim,hidden_dim ), minval=-0.15, maxval=0.15),trainable = True)
        # self.weights2 = tf.Variable(tf.random.uniform(shape=(hidden_dim,num_transitions ), minval=-0.15, maxval=0.15),trainable = True)
        self.weights2 = tf.Variable(tf.random.truncated_normal([hidden_dim,num_transitions], mean=0, stddev=1.0 / math.sqrt(num_transitions)),trainable = True)
        self.bias =tf.Variable(tf.zeros([hidden_dim]), trainable = True)
        # self.embeddings = tf.Variable(tf.random.uniform(shape=(vocab_size, embedding_dim), minval=-0.01, maxval=0.011),trainable = trainable_embeddings)
        self.embeddings = tf.Variable(tf.random.truncated_normal([vocab_size, embedding_dim], mean=0, stddev=0.01),trainable=trainable_embeddings)
        # TODO(Students) End

    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in


            the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        ems = tf.nn.embedding_lookup(self.embeddings, inputs)
        ems = tf.reshape(ems, (int(inputs.shape[0]), int(ems.shape[1] * ems.shape[2])), -1)
        neuron_calc = tf.matmul(ems,self.weights1) + self.bias
        layer1 = self._activation(neuron_calc)
        layer2 = tf.matmul(layer1, self.weights2)
        logits=layer2;
        # TODO(Students) End
        output_dict = {"logits": logits}
        if labels is not None:
            output_dict["loss"]=self.compute_loss(logits, labels)
        return output_dict

    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> object:
        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        loss = self.cross_entropy_loss(logits, labels)

        w1_r = tf.nn.l2_loss(self.weights1)
        w2_r = tf.nn.l2_loss(self.weights2)
        b_r = tf.nn.l2_loss(self.bias)
        e_r = tf.nn.l2_loss(self.embeddings)

        regularization = w1_r * self._regularization_lambda + w2_r * self._regularization_lambda \
                         + b_r * self._regularization_lambda + e_r * self._regularization_lambda;
        # TODO(Students) End
        return loss + regularization


    def cross_entropy_loss(self, logits, labels):
        # import pdb; pdb.set_trace();
        labels = tf.dtypes.cast(labels, tf.float32)
        feasible = tf.cast(labels > -1, 1)
        labels_to_consider = tf.math.multiply(labels, feasible)
        max_ = tf.expand_dims(tf.math.reduce_max(logits,axis=1),axis=1)
        stable_logits = tf.math.subtract(logits, max_)
        numerator = tf.math.exp(stable_logits)
        numerator_to_consider = tf.math.multiply(feasible,numerator)
        denominator = tf.reshape(tf.math.reduce_sum(numerator_to_consider,axis=1),(-1,1))
        softmax = tf.math.divide(numerator_to_consider,denominator);
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels_to_consider * tf.math.log(softmax+10e-19), axis=1))
        return (cross_entropy);