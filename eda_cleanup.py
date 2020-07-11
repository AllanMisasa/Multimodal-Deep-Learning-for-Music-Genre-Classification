# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:07:43 2020

@author: allan
"""

import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from itertools import chain
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

msd = pd.read_csv('D:\Datasets\FinalCogSci\MSD_subset.csv')
msd = msd[msd['Genre'].notna()]
mxm = pd.read_csv('D:\Datasets\FinalCogSci\mxm.csv', sep='\s+', header=None, names=['TrackId','mxmId','Words'])

for_pd = StringIO()

with open('D:\Datasets\FinalCogSci\mxm.csv') as temp:
    for line in temp:
        new_line = re.sub(r',', '|', line.rstrip(), count=2)
        print (new_line, file=for_pd)

for_pd.seek(0)

mxm = pd.read_csv(for_pd, sep='|', header=None, names=['TrackId','mxmId','Words'])

# Track ids for matching
mxm_trackids = pd.read_csv('D:\Datasets\FinalCogSci\mxm.csv', header = None, usecols = [0])

with open('D:\Datasets\FinalCogSci\mxm_word_list.csv', mode='r') as infile:
    reader = csv.reader(infile)
    with open('D:\Datasets\FinalCogSci\mxm_word_list_dict.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        mydict = {rows[0]:rows[1] for rows in reader}



temp= []
with open('D:\Datasets\FinalCogSci\word_list.csv', mode='r', encoding='utf-8') as file_data:
    reader = csv.DictReader(file_data, ("title", "value"))
    for i in reader:
        temp.append(i)
# bag of words in dictionary form of all words
data = {}
for i in temp:
  data[i["title"]] = i["value"]

word_list = []
for i in temp:  
    word_list.append(i["title"])

#acceptable_genres = ['jazz', 'pop', 'rock', 'metal', 'indie', 'classical', 'punk', 'electronic', 'blues', 'ambient', 'folk', 'hip hop', 'country']

acceptable_genres = ['classical', 'electronic', 'folk', 'country']

song_matches = np.intersect1d(mxm_trackids[0], msd.loc[:, ['TrackId']])

# MSD subsets with audio features and genre labels.
msd_feats_max = msd.loc[:, ['TrackId', 'Duration', 'Loudness', 'Tempo', 'TimeSignature', 'Key', 'Mode']]
msd_labels = msd.loc[:, ['TrackId', 'Genre']]

# Find all matches to minimize the dataset to ones we have features for
def matched_set(id_match, dataset):
    # new datataframe d by the 
    d = dataset[dataset['TrackId'].isin(id_match)]
    return d
            
matched_msd_full = matched_set(song_matches, msd_feats_max)
matched_msd_full.reset_index(drop=True, inplace=True)
matched_msd_labels = matched_set(song_matches, msd_labels)
matched_msd_labels.reset_index(drop=True, inplace=True)
matched_mxm = matched_set(song_matches, mxm)
matched_mxm.reset_index(drop=True, inplace=True)

def accept_genre(string, acceptables):
    rep = []
    stringa = string.split()
    for word in stringa:
        if word in acceptables:
            rep.append(word)
        else:
            pass
    return list(set(rep))
 
matched_msd_labels['Genre'] =  [accept_genre(x, acceptable_genres) for x in matched_msd_labels['Genre']]


indexNames = matched_msd_labels[matched_msd_labels['Genre'].map(len) == 0].index
matched_msd_labels.drop(indexNames , inplace=True)

song_matches2 = matched_msd_labels['TrackId']
matched_mxm = matched_set(song_matches2, mxm)
matched_msd_full = matched_set(song_matches2, matched_msd_full)
matched_msd_full.reset_index(drop=True, inplace=True)


conv_to_dict = []
split = [x.split(',') for x in matched_mxm['Words']]
for i in split:
    z = [n.split(':') for n in i]
    conv_to_dict.append(z)

flat_conv = [item for sublist in conv_to_dict for item in sublist]

def listToDict(lst):
    op = {lst[0]: int(lst[1])}
    return op


for n in conv_to_dict:
    for i in range(len(n)):
        n[i] = listToDict(n[i])


def vectorizeDict(dictionary):
    vec = DictVectorizer()
    word_feats = []
    for i in conv_to_dict:
        te = (vec.fit_transform(i).toarray())
        word_feats.append(te)
    return word_feats

word_feats = np.array(vectorizeDict(conv_to_dict))
word_feats_labels = np.column_stack((word_feats, matched_msd_labels['Genre']))


#Train_X_Tfidf = Tfidf_vect.transform(Train_X)

count_vectorizer = CountVectorizer()   
count_vect = count_vectorizer.fit(word_list)
dict_list = list(data)

def mxm_vectorizer():
    full_lyrics = []
    for i in matched_mxm.index:
        wordi = []
        bow = matched_mxm['Words'][i].split(',')
        for n in bow:
            word = int(n.split(':')[0])
            count = int(n.split(':')[1])
            unravelword = dict_list[word-1]
            replacement = [unravelword] * count
            wordi.append(replacement)
        full_lyrics.append(wordi)
        
    return full_lyrics

full_lyrics = mxm_vectorizer()
flat_lyrics = []

for i in full_lyrics:
    flat = chain.from_iterable(i)
    flat = list(flat)
    flat = ' '.join(flat)
    flat_lyrics.append(flat)

def tf_idf(flattened_lyrics):

    X_lyrics = count_vect.transform(flattened_lyrics)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_lyrics)
    X_tf_lyrics = tf_transformer.transform(X_lyrics)
    return X_tf_lyrics

X_tf = tf_idf(flat_lyrics)

def prep_data_for_class():
    msd = matched_msd_full.loc[:, ['Duration', 'Loudness', 'Tempo', 'TimeSignature', 'Key', 'Mode']]
    y = matched_msd_labels['Genre']
    y = MultiLabelBinarizer().fit_transform(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(msd) 
    return X, y

X, y = prep_data_for_class()

X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.20, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X_tf, y, test_size=0.20, random_state=42)


clf_au = OneVsRestClassifier(RandomForestClassifier())
clf_au.fit(X1_train, y1_train)
pred_au = clf_au.predict(X1_test)

clf_ly = OneVsRestClassifier(RandomForestClassifier())
clf_ly.fit(X2_train, y1_train)
pred_ly = clf_ly.predict(X2_test)

def precision_recall(y1_test, pred):
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(4):
        precision[i], recall[i], _ = precision_recall_curve(y1_test[:, i], pred[:, i])
        average_precision[i] = average_precision_score(y1_test[:, i], pred[:, i])
    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y1_test.ravel(), pred.ravel())
    average_precision["micro"] = average_precision_score(y1_test, pred, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
    
    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
    
precision_recall(y2_test, pred_ly)

joint_class = np.concatenate((pred_au, pred_ly), axis=1)
X_train, X_test, y_train, y_test = train_test_split(joint_class, y, test_size=0.20, random_state=42)
