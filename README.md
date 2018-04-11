# python_libraries
Various functions for text extraction, text embedding and neural network components

# tensorflow_modules
model.py - Contains a baseline class for a generic neural network model and a neural layer. the model class can be inherited to define a custom deep learning network
modules.py - contains classes for different neural network components (hidden layer for vanilla NN, 1d Convolutional layer, 2d Convolutional layer, LSTM, Bi-RNN, etc
Idea is to use these components in your complex models as lego blocks.

# nlp_functions
senentce_operations - various sentence operations lile splitting by white_space, converting sentences to word and character embeddings, etc. used for various nlp projects so was easier to standardize here
word_and_character_vectors - functions to load glove, fasttext and character embeddings
