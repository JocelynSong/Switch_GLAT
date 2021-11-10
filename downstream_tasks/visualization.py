# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:08:14 2021
@author: song
visual data
"""
import time
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn import manifold


def read_freq_words(filename):
    freq_words = []
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines()[: 3000]:
        freq_words.append(line.strip())
    return freq_words


def read_freq_word_embedding(emb_dict, freq_words):
    embeddings = []
    labels = []
    for word in freq_words:
        if word.lower() not in emb_dict:
            print(word)
            continue
        labels.append(word.lower())
        embeddings.append(emb_dict[word])
    return labels, embeddings


def read_aligned_word_embedding(src_dict, src_words, tgt_dict, tgt_words):
    src_embeddings = []
    src_labels = []

    tgt_embeddings = []
    tgt_labels = []
    for src, tgt in zip(src_words, tgt_words):
        if src not in src_dict:
            print(src)
            continue
        if tgt not in tgt_dict:
            print(tgt)
            continue
        src_labels.append(src)
        src_embeddings.append(src_dict[src])
        tgt_labels.append(tgt)
        tgt_embeddings.append(tgt_dict[tgt])
    return src_labels, src_embeddings, tgt_labels, tgt_embeddings


def read_word_embedding(word_file, embeb_file):
    words = open(word_file, "r", encoding="utf-8").readlines()[: 3000]
    embs = open(embeb_file, "r", encoding="utf-8").readlines()[: 3000]
    embeddings = dict()
    for word, emb in zip(words, embs):
        lemmas = emb.strip().split()
        word = word.strip()
        embedding = []
        for lemma in lemmas:
            embedding.append(float(lemma.strip()))
        embeddings[word] = np.array(embedding)
    return embeddings


def get_new_words(filename):
    f = open(filename, 'r')
    y = []
    for line in f.readlines():
        l = line.strip()
        y.append(l)
    f.close()
    return y


def get_distance(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    dist = np.sum(np.square(x1 - x2))
    print(dist)
    return dist


# ----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(length, good_words, y, X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # plt.figure(figsize=(100,100))
    # plt.scatter(X[:40, 0],X[:40, 1], c='g', marker='s', s=50)
    # plt.scatter(X[40:, 0],X[40:, 1], c='midnightblue', marker='s', s=50)
    plt.plot(X[:length, 0], X[:length, 1], 'r^')
    plt.plot(X[length:, 0], X[length:, 1], 'b*')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.05, 1.05])
    plt.show()
    '''
    plt.plot(X[:40,0],X[:40,1],'r^')
    plt.plot(X[40:,0],X[40:,1],'b*')
    '''
    exit(0)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\simhei.ttf", size=20)
    # font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\ARIALN.ttf", size=20)
    font = FontProperties(fname="/Library/Fonts/Arial.ttf", size=20)
    for i in range(39):
        plt.annotate('%s' % y[i], xy=(X[i][0], X[i][1]),
                     # textcoords='offset points',
                     ha='center', va='center', bbox=dict(boxstyle="round", fc="w", color='g'),
                     color='g', fontproperties=font)
    # font = FontProperties(fname=r"C:\\Windows\\Fonts\\times.ttf", size=15)
    font = FontProperties(fname="/Library/Fonts/Times New Roman.ttf", size=20)
    for i in range(39, 78):

        if y[i] not in good_words:
            plt.annotate('%s' % y[i], xy=(X[i][0], X[i][1]),
                         # textcoords='offset points',
                         ha='center', va='center', bbox=dict(boxstyle="round", fc="w", color='midnightblue'),
                         color='midnightblue', fontproperties=font)
        else:
            plt.annotate('%s' % y[i], xy=(X[i][0], X[i][1]),
                         # textcoords='offset points',
                         ha='center', va='center', bbox=dict(boxstyle="round", fc="w", color='olive'),
                         color='olive', fontproperties=font)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.show()


def main():
    src_word_file = "./../../visualization/words/en-de.de.txt"
    tar_word_file = "./../../visualization/words/en-de.en.txt"
    src_file = "./../../visualization/mglat_emb/freq.word.emb.de.txt"
    tar_file = "./../../visualization/mglat_emb/freq.word.emb.en.txt"

    freq_path = "./../../visualization/words"
    src_freq_file = os.path.join(freq_path, "en-de.de.txt")
    tar_freq_file = os.path.join(freq_path, "en-de.en.txt")

    src_freq_words = read_freq_words(src_freq_file)
    tar_freq_words = read_freq_words(tar_freq_file)
    print(len(src_freq_words))
    print(len(tar_freq_words))

    src_emb_dict = read_word_embedding(src_word_file, src_file)
    tgt_emb_dict = read_word_embedding(tar_word_file, tar_file)

    src_words, src_freq_emb, tar_words, tar_freq_emb = read_aligned_word_embedding(src_emb_dict, src_freq_words,
                                                                                   tgt_emb_dict, tar_freq_words)

    print("src len word: {}".format(len(src_words)))
    print("tgt len word: {}".format(len(tar_words)))

    features = src_freq_emb
    labels = src_words

    bad_words = ['course', 'point', "year", "question", "quality", "business", "agreement",
                  'problem', 'company', 'position', 'hope', 'action', 'law', 'current',
                  'health', 'research', 'resolution', "policy", 'time', 'legal', 'point']
    # noise_words = ['people', 'support', 'report', 'hotel', 'right', 'order', 'good', 'year', 'political',
    #                'public', 'international', 'social', 'great', 'believe', 'future', 'today', "process",
    #                "service", "human", "debate", "proposal"]
    good_words = []
    length = len(features)
    for i, word in enumerate(tar_words):
        labels.append(word)
        features.append(tar_freq_emb[i])
        # features.append(src_freq_emb[i] + 0.5)
        # if src_words[i] in bad_words:
        #     features.append(tar_freq_emb[i])
        # else:
        #     features.append(src_freq_emb[i] + 0.01)

    features = np.array(features)

    n_samples, n_features = features.shape
    n_neighbors = 30
    print("feature shape:%d, %d\n" % (n_samples, n_features))
    print("length of labels:%d\n" % len(labels))

    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time.time()
    x_tsne = tsne.fit_transform(features)


    print("Starting draw picture\n")
    plt.figure(figsize=(12, 8))
    plt.xticks([])
    plt.yticks([])
    # plt.title('Baseline Model',color='dimgrey')
    # plt.xlabel('BWE')
    plot_embedding(length, good_words, labels, x_tsne, "t-SNE embedding of the digits (time %.2fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
