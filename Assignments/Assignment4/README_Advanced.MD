#Advanced Model
The advanced model contain 6 setting of the following

#CNN Layers  for feature extractions
The kernel size of 2,3,4 captures micro, a bit macro, and macro features of the sentence. The use of kernel size as 32 is selected after multiples test of 32, 64, and 128 Kernel sizes. The maxpool is a global maxpool as it choses 1 cell from a list of cells. The ReLU is a leaky relu as it is a better choice over vanilla ReLU.
A)

1. CNN Layer of Kernel size (2, 32)
2. Leaky ReLU
3. MaxPool

B)

1. CNN Layer of Kernel size (3, 32)
2. Leaky ReLU
3. MaxPool

C)

1. CNN Layer of Kernel size (4, 32)
2. Leaky ReLU
3. MaxPool

D)

1. CNN Layer of Kernel size (2, 32)
2. Leaky ReLU
3. MaxPool

E)

1. CNN Layer of Kernel size (3, 32)
2. Leaky ReLU
3. MaxPool

F)
1. CNN Layer of Kernel size (4, 32)
2. Leaky ReLU
3. MaxPool



G) CONCAT (A,B,C,D,E,F)  The idea for this stems from the paper "Relation Classification via Convolutional Deep Neural Network".


H) Tanh(G) As the paper suggests tanh on the sentence features extracted above from the CNN proves better.

I) Attention(Word Embedding) 

J) Attention(Pos Embeddding) Using the attention on the word embedding and the pos tag embedding and concatenating it with the sentence features gives better results.

#Classification Layers for Classifications

K) Concat(H, I, J)

L) Flatten K for Linear Layer. This is done to get a linear layer for classification.

M) Dropout(0.4) L to avoid overfitting. AS we have seen in the previous assignments, dropout of 0.4 seems optimal as it avoids overfitting.

N) Linear Layer for classification. This is the last layer similar to the decoer of TA.








