#rom __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import collections
import math
import os, sys
import random
import zipfile

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
import loss_func as tf_func

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pickle
from collections import namedtuple


Word2Vec = namedtuple('Word2Vec', ['train_inputs', 'train_labels', 'loss', 'optimizer', 'global_step',
                                   'embeddings', 'normalized_embeddings', 'valid_embeddings', 'similarity',
                                   'saver', 'summary', 'summary_writer'])


def maybe_create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print("Created a path: %s" % (path))

def load_pretrained_model(sess, model, pretrained_model_path):
    if not os.path.exists(filename):
        print("Missing pre-trained model: [%s]" % (pretrained_model_path))
        return

    ckpt = tf.train.get_checkpoint_state(pretrained_model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)


def train(sess, model, data, dictionary, batch_size, num_skips, skip_window,
          max_num_steps, checkpoint_step, loss_model):
    print("Training begins")
    average_loss_step = max(checkpoint_step / 10, 100)

    average_loss = 0
    for step in xrange(max_num_steps):
        batch_inputs, batch_labels = generate_batch(data, batch_size, num_skips, skip_window)
        feed_dict = {model.train_inputs.name: batch_inputs, model.train_labels.name: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        # _, loss_val, summary = sess.run([model.optimizer, model.loss, model.summary], feed_dict=feed_dict)
        _, loss_val = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % average_loss_step == 0:
            if step > 0:
                average_loss /= average_loss_step
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0
            # model.summary_writer.add_summary(summary, model.global_step.eval())
            # model.summary_writer.flush()

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % checkpoint_step == 0:
            sim = model.similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
            # chkpt_path = os.path.join(checkpoint_model_path, 'w2v_%s.cpkt'%(loss_model))
            # model.saver.save(sess, chkpt_path, global_step=model.global_step.eval())

    # model.summary_writer.close()

    # Saving the final embedding to a file
    final_embeddings = model.normalized_embeddings.eval()

    return final_embeddings


def build_model(sess, graph, loss_model):
    """
    Builds a tensor graph model
    """
    model = None
    with graph.as_default():
        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Input data.
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            global_step = tf.Variable(0, trainable=False)

            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            print("Embeddings", embeddings)
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            sm_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))

            # Get context embeddings from lables
            true_w = tf.nn.embedding_lookup(sm_weights, train_labels)
            true_w = tf.reshape(true_w, [-1, embedding_size])

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        loss_model = 'cross_entropy';
        if loss_model == 'cross_entropy':
            loss = tf.reduce_mean(tf_func.cross_entropy_loss(embed, true_w))
        else:
            # sample negative examples with unigram probability
            sample = np.random.choice(vocabulary_size, num_sampled, p=unigram_prob, replace=False)

            loss = tf.reduce_mean(tf_func.nce_loss_og(embed, nce_weights, nce_biases, train_labels, sample, unigram_prob))

        # tf.summary.scalar('loss', loss)

        # Construct the SGD optimizer using a learning rate of 1.0.
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss, global_step=global_step)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm

        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        saver = tf.train.Saver(tf.global_variables())

        # Save summary
        # summary = tf.summary.merge_all()
        # summary_writer = tf.summary.FileWriter(summary_path + '/summary', sess.graph)
        summary = None
        summary_writer = None

        tf.global_variables_initializer().run()
        print("Initialized")

    model = Word2Vec(train_inputs, train_labels, loss, optimizer, global_step, embeddings,
                     normalized_embeddings, valid_embeddings, similarity, saver, summary, summary_writer)

    return model





def generate_batch(data, batch_size, num_skips, skip_window):
    """
  Write the code generate a training batch

  @data_index: the index of a word. You can access a word using data[data_index]
  @batch_size: the number of instances in one batch
  @num_skips: the number of samples you want to draw in a window
            (In the below example, it was 2)
  @skip_windows: decides how many words to consider left and right from a context word.
                (So, skip_windows*2+1 = window_size)

  batch will contain word ids for context words. Dimension is [batch_size].
  labels will contain word ids for predicting(target) words. Dimension is [batch_size, 1].


  """

    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    start_of_window = 0;
    end_of_window = start_of_window + skip_window * 2;
    batch_size_ctr = 0;

    while batch_size_ctr < batch_size:
        center_of_window = start_of_window + skip_window;
        context_word = data[center_of_window]
        iter = num_skips;
        flip = 1;
        while iter > 0:
            if flip == -1:
                labels[batch_size_ctr] = data[center_of_window + ((num_skips - iter) * flip) - 1];
                batch[batch_size_ctr] = context_word;
                batch_size_ctr = batch_size_ctr + 1;
            else:
                labels[batch_size_ctr] = data[center_of_window + ((num_skips - iter) * flip) + 1];
                batch[batch_size_ctr] = context_word;
                batch_size_ctr = batch_size_ctr + 1;
            flip = flip * -1;
            iter -= 1;

        start_of_window = start_of_window + 1;
    return batch, labels




def maybe_download(filename, expected_bytes):
    # Download a file if not present, and make sure it's the right size.
    if not os.path.exists(filename):
        print('Downloading %s' % (url + filename))
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


# Read the data into a list of strings.
def read_data(filename):
    # Extract the first file enclosed in a zip file as a list of words
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    # print(count[:10]) === Count stores common word and thier count
    # [['UNK', -1],
    # ('the', 1061396),
    # ('of', 593677),
    # ('and', 416629),
    # ('one', 411764),
    # ('in', 372201),
    # ('a', 325873),
    # ('to', 316376),
    # ('zero', 264975),
    # ('nine', 250430)]
    # print(len(words)) === 17005207
    # print(len(count)) ===  100000
    dictionary = dict()

    for word, _ in count:
        dictionary[word] = len(dictionary)
    # print(dictionary['UNK']) === 0 ; dictionary stores the index of the most common word
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count #Unknown count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

if __name__ == '__main__':

    loss_model = 'cross_entropy'
    if len(sys.argv) > 1:
        if sys.argv[1] == 'nce':
            loss_model = 'nce'
####################################################################################
    # Step 1: Download the data.
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download('text8.zip', 31344016)

    words = read_data(filename)
    print('Data size', len(words))
    vocabulary_size = 100000

    data, count, dictionary, reverse_dictionary = build_dataset(words)

    unigram_cnt = [c for w, c in count]
    total = sum(unigram_cnt)
    unigram_prob = [c * 1.0 / total for c in unigram_cnt] # Probability of word occurence

    data_index = 0

    batch, labels = generate_batch(data, batch_size=8, num_skips=2, skip_window=1)
    #batch_og, labels_og = generate_batch_og(data, batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

    #print(len(batch), " OG: " , len(batch_og))
    #print(len(labels), " OG: ", len(labels_og))

    #for i in range(8):
    #    print(batch_og[i], reverse_dictionary[batch_og[i]], '->', labels_og[i, 0], reverse_dictionary[labels_og[i, 0]])

    batch_size = 32
    embedding_size = 32  # Dimension of the embedding vector.
    skip_window = 16  # How many words to consider left and right.
    num_skips = 8  # How many times to reuse an input to generate a label.

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 64  # Number of negative examples to sample.
    print("Valid examples: " , valid_examples)
    # summary_path = './summary_%s'%(loss_model)
    pretrained_model_path = './pretrained/'

    checkpoint_model_path = './checkpoints_%s/' % (loss_model)
    model_path = './models'

    # maximum training step
    max_num_steps = 200001
    checkpoint_step = 50000

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:

        ####################################################################################
        # Step 4: Build and train a skip-gram model.
        model = build_model(sess, graph, loss_model)

        # You must start with the pretrained model.
        # If you want to resume from your checkpoints, change this path name

        load_pretrained_model(sess, model, pretrained_model_path)
        print("Model loading complete")
        ####################################################################################
        # Step 6: Begin training.
        maybe_create_path(checkpoint_model_path)
        embeddings = train(sess, model, data, dictionary, batch_size, num_skips, skip_window,
                           max_num_steps, checkpoint_step, loss_model)

        ####################################################################################
        # Step 7: Save the trained model.
        trained_steps = model.global_step.eval()

        maybe_create_path(model_path)
        model_filepath = os.path.join(model_path, 'word2vec_%s.model' % (loss_model))
        print("Saving word2vec model as [%s]" % (model_filepath))
        pickle.dump([dictionary, trained_steps, embeddings], open(model_filepath, 'wb'))

