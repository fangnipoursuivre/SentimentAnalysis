"""
Created on May 1st, 2020 - München

Sentiment Analysis of Health Authority Feedback

@author: Dr. FANG Ni
"""

# Convolutional Neural Network

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold

# ------------------------------------------------------------------#
trainfile = 'sentences_with_sentiment.xlsx'  # training data file   #
labelfile = 'label.pickle'                   # label value          #
# ------------------------------------------------------------------#

def load_train_data(trainfile):
    print(datetime.today().strftime('%Y-%m-%d %H:%M:%S'), ': Training data has been loaded.')
    df = pd.read_excel(trainfile, sheet_name='Sheet1', encoding='utf-8-sig')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def data_summary():
    print('------ Training Data Summary ------')
    print('Total number of words: %d' % df['Sentence'].apply(lambda x: len(x.split(' '))).sum())
    print('Total number of instance is %d with the ratio being 1:%d:%d for NEGATIVE, NEUTRAL and POSITIVE.'
          % (df.shape[0], round(len(df[df['Neutral'] == 1]) / len(df[df['Negative'] == 1])),
             round(len(df[df['Positive'] == 1]) / len(df[df['Negative'] == 1]))))


def process_text():
    # tokenization
    df['Tokenized'] = [entry.lower() for entry in df['Sentence']]
    df['Tokenized'] = [word_tokenize(entry) for entry in df['Tokenized']]

    # remove stop-word & non-alphabet, perform words stemming/lemmenting
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['adj'] = wn.ADJ
    tag_map['verb'] = wn.VERB
    tag_map['adv'] = wn.ADV

    for index, entry in enumerate(df['Tokenized']):
        final_words = []
        word_Lemmatized = WordNetLemmatizer()
        # provide the words with tag
        for word, tag in pos_tag(entry):
            # check for stop-word and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                final_words.append(word_final)
        df.loc[index, 'Tokenized'] = str(final_words)
    return df


def feature_label():
    X = df['Tokenized']

    df.loc[df['Positive'] == 1, 'Sentiment'] = 2  # Positive
    df.loc[df['Neutral'] == 1, 'Sentiment'] = 0  # Neutral
    df.loc[df['Negative'] == 1, 'Sentiment'] = 1  # Negative

    # store label column
    with open(labelfile, 'wb') as abc:
        pickle.dump(df['Sentiment'], abc)

    TFIDF_vect = TfidfVectorizer(max_features=5000)
    TFIDF_vect.fit(df['Tokenized'])

    feature = TFIDF_vect.transform(X).toarray()
    label = df['Sentiment']
    return feature, label


def cnn_model(n, b, e, k):
    skf = StratifiedKFold(n_splits=n)

    feature, label = feature_label()
    feature = np.expand_dims(feature, -1)  # reshape (266, 1033) to (266, 1033, 1)
    print('\n')
    print(datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
          ': Convolution Neural Network to perform multiclass text classification training through K-fold cross validation.')

    with open(labelfile, 'rb') as abc:
        dx = pickle.load(abc)

    # number of classes
    nClasses = len(dx.unique())


    for train_index, test_index in skf.split(feature, label):
        train_X, test_X = feature[train_index], feature[test_index]
        train_y, test_y = label[train_index], label[test_index]

        # Change the labels from categorical to one-hot encoding
        train_y_one_hot = to_categorical(train_y)
        test_y_one_hot = to_categorical(test_y)

        # convolutional neural network architecture
        cnn = Sequential()
        cnn.add(Conv1D(32, kernel_size=(k), activation='relu', input_shape=(feature.shape[1], 1)))
        cnn.add(MaxPooling1D((1)))
        cnn.add(Flatten())
        cnn.add(Dense(128, activation='relu'))
        cnn.add(Dense(nClasses, activation='sigmoid'))

        # compile model
        cnn.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        print(cnn.summary())

        # train model
        cnn.fit(train_X, train_y_one_hot, batch_size=b, epochs=e, verbose=1,validation_data=(test_X, test_y_one_hot))

        # evaluation
        test_eval = cnn.evaluate(test_X, test_y_one_hot, verbose=0)
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])


def main():
    data_summary()
    process_text()
    cnn_model(4, 64, 20, 6)  # k-fold, batch_size, epochs, kernel_size


if __name__ == '__main__':
    df = load_train_data(trainfile)
    main()
