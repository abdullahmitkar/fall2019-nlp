# inbuilt lib imports:
from typing import Dict
import math

# external libs
import tensorflow as tf
from tensorflow.keras import models, layers
import pdb

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
        return tf.pow(vector,3)
        raise NotImplementedError
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
                 activation_name: str = "cubic",
                 embeddings: list=[]) -> None:
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

        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._num_tokens = num_tokens
        self._hidden_dim = hidden_dim
        self._num_transitions = num_transitions
        self._regularization_lambda = regularization_lambda
        self._trainable_embeddings = trainable_embeddings
        self._activation_name = activation_name


        # Trainable Variables

        # TODO(Students) Start
        # initializer1 =
        r = tf.initializers.TruncatedNormal(-0.01,0.01)
        initializer2 = tf.initializers.RandomNormal()
        # r = tf.random_uniform_initializer(-0.01,0.01)
        one\
            = tf.initializers.Ones()


        self.embeddings = tf.Variable(r([self._vocab_size,self._embedding_dim]),trainable = self._trainable_embeddings)
        self.w1 = tf.Variable(r([self._hidden_dim,self._embedding_dim*self._num_tokens]),trainable = True)
        self.b1 = tf.Variable(tf.zeros([self._hidden_dim]),trainable = True)
        self.w2 = tf.Variable(r([self._num_transitions,self._hidden_dim]),trainable = True)
        self.b2 = tf.Variable(tf.zeros([self._num_transitions]),trainable = True)

        # self.W_p = tf.Variable(initializer([self._embedding_dim*18,self._hidden_dim]))
        # self.W_l = tf.Variable(initializer([self._embedding_dim*12,self._hidden_dim]))
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
            should be made in the given configuration.

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
        # print(self._vocab_size)
        # print(self._embedding_dim,self._hidden_dim)
        # print(inputs)
        # print(labels)
        # print(labels.shape)

        embed = tf.nn.embedding_lookup(self.embeddings,inputs)
        layer1 = tf.reshape(embed,[embed.shape[0],embed.shape[1]*embed.shape[2]])
        layer2 = tf.add(tf.matmul(layer1,tf.transpose(self.w1)),self.b1)
        layer2 = self._activation(layer2)
        # layer2 = tf.nn.dropout(layer2,0.5)
        logits = tf.add(tf.matmul(layer2,tf.transpose(self.w2)),self.b2)
        # print(logits)
        # print(self.w2)
        # x = tf.matmul(layer2,tf.transpose(self.w2))
        # logits = tf.nn.softmax(x)

        # pdb.set_trace()
        # TODO(Students) End
        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict

    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:
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
        # loss =


        # mask0 = tf.where(labels == 0, True, False)
        # mask1 = tf.where(labels == 1, True, False)
        # print(labels.shape)

        mask = tf.where(labels == -1, False, True)
        valid_logits = tf.ragged.boolean_mask(logits,mask)
        valid_labels = tf.ragged.boolean_mask(labels,mask)
        deno = tf.math.reduce_sum(tf.math.exp(valid_logits),axis = 1)
        nume = tf.math.exp(valid_logits)
        softmax = nume/tf.reshape(deno,(-1,1))
        valid_labels = tf.dtypes.cast(valid_labels,tf.float32)
        inter = valid_labels * -tf.math.log(softmax+0.0000000000000001)
        loss = tf.math.reduce_mean(tf.math.reduce_sum(inter,axis = 1),axis=0)
        regularization = self._regularization_lambda*(tf.nn.l2_loss(self.w1)+tf.nn.l2_loss(self.w2)+tf.nn.l2_loss(self.b1)+tf.nn.l2_loss(self.b2)+tf.nn.l2_loss(self.embeddings))

        # mask = tf.where(labels == -1, True, False)
        # # print(labels)
        # valid_labels = tf.where(mask,0,labels)
        # # print(logits)
        # valid_logits = tf.where(mask,float("-inf"),logits)
        # # print(valid_logits)
        #
        # softmax = tf.nn.softmax(valid_logits)
        # valid_labels = tf.dtypes.cast(valid_labels,tf.float32)
        #
        # inter = valid_labels * -tf.math.log(softmax+0.0000000000000001)
        # loss = tf.math.reduce_mean(tf.math.reduce_sum(inter,axis = 1),axis=0)
        # regularization = self._regularization_lambda*(tf.nn.l2_loss(self.w1)+tf.nn.l2_loss(self.w2)+tf.nn.l2_loss(self.b1)+tf.nn.l2_loss(self.b2)+tf.nn.l2_loss(self.embeddings))






        # y = tf.pad(y, [[0, 0], [0, self.ps.len_max_input - s], [0, 0]])
        # print(valid_labels)
        # print(labels,mask1,mask0)
        # print(logits)




        # valid_logits_0 = tf.ragged.boolean_mask(logits,mask0)
        # valid_logits_1 = tf.ragged.boolean_mask(logits,mask1)

        # print(valid_logits_0.shape)
        # print(valid_logits_1)

        # true_label_0 = tf.zeros(valid_logits_0.shape[0])
        # true_label_1 = tf.ones(valid_logits_1.shape[0])
        # true_label = tf.concat([true_label_0,true_label_1],0)


        # print(true_label.shape)
        # true_label = tf.reshape(true_label,[true_label.shape[0],1])

        # pred_label = tf.concat([valid_logits_0,valid_logits_1],0)
        # pred_label = tf.reshape(pred_label,[pred_label.shape[0],1])
        # print(true_label,pred_label)
        # print(true_label)
        # loss = tf.nn.softmax_cross_entropy_with_logits(true_label,pred_label)

        # tf.reduce_mean(-tf.reduce_sum(y_true_tf * tf.math.log(pred_label), axis=1))

        # loss = -(tf.reduce_mean(tf.math.log(1-valid_logits_0+0.0000000000001))+tf.reduce_mean(tf.math.log(valid_logits_1+0.0000000001)))/2

        # valid_logits = tf.athevalid_logits - max(valid_logits)

        # softmax = tf.sparse.softmax(valid_logits)


        # pdb.set_trace()





        # pdb.set_trace()
        # print(tf.math.log(softmax+0.0000000000000001),tf.math.log(1-(softmax+0.00000000000000001)))
        # loss = tf.nn.softmax_cross_entropy_with_logits(true_label,pred_label)
        # print(true_label *-tf.math.log(pred_label+0.000000000000001))
        # valid_labels *-tf.math.log(softmax+0.0000000000000001)
        # pdb.set_trace()


        # pdb.set_trace()
        # +(1-valid_labels)*-tf.math.log((1-(softmax))+0.00000000000001)

        # loss = -tf.reduce_mean(tf.math.log(valid_logits_1+0.0000000001))
        # pdb.set_trace()
        # regularization = 0
        # loss =
        # TODO(Students) End

        return loss + regularization
