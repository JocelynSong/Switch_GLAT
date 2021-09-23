import os
import sys
import numpy as np


def extract_word_net(input_file):
    lines = open(input_file, "r", encoding="utf-8").readlines()[1: ]
    word_dict = dict()
    for line in lines:
        lemmas = line.strip().split("\t")
        if len(lemmas) != 3:
            continue
        word_id = lemmas[0].strip()
        word = lemmas[2].strip()
        if word_id not in word_dict:
            word_dict[word_id] = []
        word_dict[word_id].append(word)
    return word_dict


def write_word_to_file(file_path, word_pairs, flag):
    fw = open(file_path, "w", encoding="utf-8")
    for pair in word_pairs:
        fw.write(pair[flag].strip() + "\n")
    fw.close()


def get_word_embedding(emb_line):
    emb = []
    lemmas = emb_line.strip().split()
    for lemma in lemmas:
        emb.append(float(lemma.strip()))
    return np.array(emb)


def load_word_embedding(word_file, embedding_file):
    words = open(word_file, "r", encoding="utf-8").readlines()
    embeds = np.loadtxt(embedding_file)
    embed_dict = {}
    for word_line, emb in zip(words, embeds):
        word = word_line.strip()
        # emb = get_word_embedding(emb_line)
        embed_dict[word] = emb
    return embed_dict


def get_line_embedding(line, embed_dict):
    lemmas = line.strip().split()
    embedding = []
    for lemma in lemmas:
        if lemma.strip() in embed_dict:
            embedding.append(embed_dict[lemma.strip()])
    if len(embedding) == 0:
        embedding.append(embed_dict["<unk>"])
    embedding = np.max(np.array(embedding), axis=0)
    return embedding


def get_word_bpe_embedding(word_file, embed_dict):
    lines = open(word_file, "r", encoding="utf-8").readlines()
    embed_matrix = []

    for line in lines:
        embedding = get_line_embedding(line, embed_dict)
        embed_matrix.append(embedding)
    return np.array(embed_matrix)


def normalize_maxtrix(matrix):
    length = np.sqrt(np.sum(np.dot(matrix, np.transpose(matrix)), keepdims=True))
    return matrix / length


if __name__ == "__main__":
    data_path = "./downstream_tasks/word_induction"
    embed_path = "./embeds"
    vocab_file = os.path.join(embed_path, "word.txt")
    embedding_file = os.path.join(embed_path, "mglat_embeds.txt")

    embed_dict = load_word_embedding(vocab_file, embedding_file)

    print("First en-de")

    english_bpe = os.path.join(data_path, "en-de.en.bpe")
    german_bpe = os.path.join(data_path, "en-de.de.bpe")

    en_embeds = get_word_bpe_embedding(english_bpe, embed_dict)
    de_embeds = get_word_bpe_embedding(german_bpe, embed_dict)
    en_embeds = normalize_maxtrix(en_embeds)
    de_embeds = normalize_maxtrix(de_embeds)

    indexes = np.argmax(np.dot(en_embeds, np.transpose(de_embeds)), axis=1)
    num = 0
    for i, s in enumerate(indexes):
        if i == s:
            num += 1
    acc = float(num) / len(en_embeds)
    print("En-de ACC: %f" % acc)

    indexes = np.argmax(np.dot(de_embeds, np.transpose(en_embeds)), axis=1)
    num = 0
    for i, s in enumerate(indexes):
        if i == s:
            num += 1
    acc = float(num) / len(en_embeds)
    print("De-en ACC: %f" % acc)

    print("Next en-fr")

    english_bpe = os.path.join(data_path, "en-fr.en.bpe")
    german_bpe = os.path.join(data_path, "en-fr.fr.bpe")

    en_embeds = get_word_bpe_embedding(english_bpe, embed_dict)
    de_embeds = get_word_bpe_embedding(german_bpe, embed_dict)
    en_embeds = normalize_maxtrix(en_embeds)
    de_embeds = normalize_maxtrix(de_embeds)

    indexes = np.argmax(np.dot(en_embeds, np.transpose(de_embeds)), axis=1)
    num = 0
    for i, s in enumerate(indexes):
        if i == s:
            num += 1
    acc = float(num) / len(en_embeds)
    print("En-fr ACC: %f" % acc)

    indexes = np.argmax(np.dot(de_embeds, np.transpose(en_embeds)), axis=1)
    num = 0
    for i, s in enumerate(indexes):
        if i == s:
            num += 1
    acc = float(num) / len(en_embeds)
    print("Fr-en ACC: %f" % acc)



