"""
Created on May 1st, 2020 - München

Sentiment Analysis of Health Authority Feedback

@author: Dr. FANG Ni
"""

# SVM Model

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
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# ------------------------------------------------------------------#
trainfile = 'sentences_with_sentiment.xlsx'   # training data file  #
labelfile = 'label.pickle'                    # label value         #
jobfile = "test.txt"                          # job file            #
modelfile = 'SVM.pickle'                      # trained SVM model   #
tfidffile = 'TFIDF.pickle'                    # tfidf file          #
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
            if word not in stopwords.words('english') and word.isalpha() and len(word) > 1:
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

    # compute TF-IDF score
    TFIDF_vect = TfidfVectorizer(max_features=5000, sublinear_tf=True, use_idf=True)

    # define feature & lable
    feature = TFIDF_vect.fit_transform(X).toarray()
    label = df['Sentiment']
    return feature, label

    # save tfidf-vec
    with open(tfidffile, 'wb') as abc:
        pickle.dump(TFIDF_vect.vocabulary_, abc)


def svm_model(n):
    # k-fold cross validation split
    skf = StratifiedKFold(n_splits=n)

    acc = []

    feature, label = feature_label()
    print('\n')
    print(datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
          ': SVM model to perform multi-class text classification training through K-fold cross validation.')
    print('\n')
    print('0.0 = Neutral, 1.0 = Negative, 2.0 = Positive')

    for train_index, test_index in skf.split(feature, label):
        train_X, test_X = feature[train_index], feature[test_index]
        train_y, test_y = label[train_index], label[test_index]

        smote = SMOTE('minority')

        X_sm, y_sm = smote.fit_sample(train_X, train_y)

        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        SVM.fit(X_sm, y_sm)
        predictions = SVM.predict(test_X)
        acc.append(accuracy_score(test_y, predictions))
        print(classification_report(test_y, predictions))
    print('Average Accuracy:', '{0:.2f}%'.format(np.mean(acc) * 100))

    # save model to disk
    with open(modelfile, 'wb') as abc:
        pickle.dump(SVM, abc)
    print('\n')
    print('%s : SVM model has been saved to the local directory as %s.' % (
    datetime.today().strftime('%Y-%m-%d %H:%M:%S'), modelfile))


def load_job_data(jobfile):
    # load jobfile
    fileHandler = open(jobfile, 'r')
    # Get list of all lines in file
    lisOflines = fileHandler.readlines()
    # Close file
    fileHandler.close()
    df = pd.DataFrame(columns=['Sentence'])

    n = 0
    for line in lisOflines:
        if line != '\n':
            df.loc[n] = line
            n += 1
    return df


def TFIDF_analysis(n):
    df['TFIDF Score'] = df['Tokenized']
    df['IDF Value'] = df['Tokenized']

    for index, entry in enumerate(df['Tokenized']):
        text = [df.iloc[index, 1]]
        vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True, use_idf=True)
        X = vectorizer.fit_transform(text)
        feature_array = vectorizer.get_feature_names()
        df.iloc[index, 2] = sorted(list(zip(vectorizer.get_feature_names(), np.around(X.sum(0).getA1(), decimals=4))),
                                   key=lambda x: x[1], reverse=True)[:n]
        df.iloc[index, 3] = sorted(list(zip(feature_array, vectorizer.idf_, )), key=lambda x: x[1], reverse=True)[:n]
    return df


def prediction(modelfile):
    # load trained model from the local directory
    with open(modelfile, 'rb') as abc:
        load_model = pickle.load(abc)
        print('\n')
        print(datetime.today().strftime('%Y-%m-%d %H:%M:%S'), ': SVM model has been loaded for prediction task.')

    # define predictor
    X = df['Tokenized']

    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(vocabulary=pickle.load(open(tfidffile, 'rb')))
    predictor = transformer.fit_transform(loaded_vec.fit_transform(X)).toarray()

    # perform prediction
    result = load_model.predict(predictor)

    with open(labelfile, 'rb') as abc:
        dx = pickle.load(abc)

    df['Prediction'] = 0
    for x in range(df.shape[0]):
        if result[x] == max(dx):
            df.iloc[x, df.columns.get_loc('Prediction')] = 'Positive'
        if result[x] == min(dx):
            df.iloc[x, df.columns.get_loc('Prediction')] = 'Neutral'
        if result[x] < max(dx) and result[x] > min(dx):
            df.iloc[x, df.columns.get_loc('Prediction')] = 'Negative'

    # finalise prediction report
    df.drop(['Tokenized', 'IDF Value'], axis=1, inplace=True)
    print('\n')
    print(datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
          ': Prediction by SVM is done, generating prediction and analysis report as follows.')
    print('\n')
    pd.set_option('display.width', 3600)
    pd.set_option('display.max_columns',10)
    print(df)


def training():
    data_summary()
    process_text()
    svm_model(4)  # value of k-fold cross-validation


def prediction_svm():
    process_text()
    TFIDF_analysis(3)  # number of top key words w.r.t.TF-IDF score
    prediction(modelfile)


if __name__ == '__main__':
    df = load_train_data(trainfile)
    training()
    df = load_job_data(jobfile)
    prediction_svm()
