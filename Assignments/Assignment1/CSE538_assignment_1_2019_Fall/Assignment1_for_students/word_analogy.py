import os
import pickle
import numpy as np
from scipy import spatial
import sys


model_path = './models/'
#loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

'''Extract word pairs from word_analogy_dev.txt and store in word_analogy_{cost}.txt'''
with open('word_analogy_dev.txt') as read_file:
    with open('word_analogy_cross_entropy_0508.txt', 'w') as out_file:
        for line in read_file:
            line = line.strip()
            left_line = line.split('||')[0]
            left_word_list = left_line.split(',')
            left_avg = []
            for combo in left_word_list:
                temp_combo = combo.split(':')
                vectA = embeddings[dictionary[temp_combo[0].strip('"')]]
                vectB = embeddings[dictionary[temp_combo[1].strip('"')]]
                left_avg.append(np.subtract(vectA, vectB))

            left_avg = np.mean(left_avg, axis=0)

            tmp_line = line.split('||')[1]
            word_list = tmp_line.split(',')
            '''Values to calculate max and min for cosine distance'''
            max_val = -99999999
            min_val = 99999999
            max_pair = ""
            min_pair = ""
            for word_pair in word_list:
                temp_pair = word_pair.split(':')
                vect1 = embeddings[dictionary[temp_pair[0].strip('"')]]
                vect2 = embeddings[dictionary[temp_pair[1].strip('"')]]
                tmp_diff = np.subtract(vect1,vect2)
                word_similarity = 1 - spatial.distance.cosine(left_avg, tmp_diff)

                '''Calculating least and most illustrative pair'''
                if word_similarity < min_val:
                    min_val = word_similarity
                    min_pair = word_pair
                if word_similarity > max_val:
                    max_val = word_similarity
                    max_pair = word_pair
            '''Writing the ouput in result file'''
            output_line = tmp_line.replace(",", " ")+" "+min_pair+" "+max_pair+"\n"
            out_file.write(output_line)