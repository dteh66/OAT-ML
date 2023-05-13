'''
Written by Austin Walters
Last Edit: October 24, 2019
For use on austingwalters.com

A CNN  to classify a sentence as one 
of the common sentance types:
Question, Statement, Command, Exclamation

It utilizes a 2-layer convolutional network

Heavily Inspired by Keras Examples: 
https://github.com/keras-team/keras
'''

from __future__ import print_function

import os
import sys

import numpy as np
import keras

from sentence_types import load_encoded_data
from sentence_types import encode_data, encode_phrases, import_embedding
from sentence_types import get_custom_test_comments

from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D

# from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

def classify(sentences, model, model_name, embedding_name, load_model_flag, maxlen, batch_size, pos_tags_flag, max_words, num_classes):
    # Use can load a different model if desired
    # model_name      = "models/2cnndense"
    # embedding_name  = "data/default"
    # load_model_flag = False
    # arguments       = sys.argv[1:len(sys.argv)]
    # if len(arguments) == 1:
        # model_name = arguments[0]
        # load_model_flag = os.path.isfile(model_name+".json")
    print(model_name)
    print("Load Model?", (load_model_flag))

    # Model configuration
    maxlen = 300
    batch_size = 64

    # Add parts-of-speech to data
    # pos_tags_flag = True

    word_encoding, category_encoding = import_embedding(embedding_name)

    # max_words   = len(word_encoding) + 1
    # num_classes = np.max(y_train) + 1

    # _, x_test, _, y_test = encode_data(sentences, test_comments_category,
    #                                 data_split=0.0,
    #                                 embedding_name=embedding_name,
    #                                 add_pos_tags_flag=pos_tags_flag)
    encoded_comments, word_encoding, word_decoding = encode_phrases(sentences, word_encoding=word_encoding, add_pos_tags_flag=False)
    encoded_comments = pad_sequences(encoded_comments, maxlen=maxlen)

    # Show predictions
    predictions = model.predict(encoded_comments, batch_size=batch_size, verbose=1)
    return predictions
    # real = []
    # test = []
    # for i in range(0, len(predictions)):
    #     real_label      = y_test[i].argmax(axis=0)
    #     predicted_label = predictions[i].argmax(axis=0)    
    #     real.append(real_label)
    #     test.append(predicted_label)

    #     if real_label != predicted_label:
    #         print("\n------- Incorrectly Labeled ----------")
    #         print("Predicted", predicted_label,
    #             "-", real_label, "real")
    #         print(test_comments[i])
    #         print("--------------------------------------\n")

    # print("Predictions")
    # print("Real", real)
    # print("Test", test)
