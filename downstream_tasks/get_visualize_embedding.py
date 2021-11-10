import numpy as np


def load_word_embedding(word_file, embedding_file):
    words = open(word_file, "r", encoding="utf-8").readlines()
    embeds = np.loadtxt(embedding_file)
    embed_dict = {}
    for word_line, emb in zip(words, embeds):
        word = word_line.strip()
        # emb = get_word_embedding(emb_line)
        embed_dict[word] = emb
    return embed_dict


def get_bpe_word_embedding(embed_dict, bpe_file, save_file):
    words = open(bpe_file, "r", encoding="utf-8").readlines()

    new_vec = []
    for word in words:
        subs = word.strip().split()
        vec = []
        for sub in subs:
            if sub not in embed_dict:
                continue
            vec.append(embed_dict[sub])
        if len(vec) == 0:
            new_vec.append(embed_dict["<unk>"])
        else:
            new_vec.append(np.max(vec, axis=0))
    new_vec = np.array(new_vec)
    np.savetxt(save_file, new_vec)


if __name__ == "__main__":
    embeding_file = "embeds.txt"
    word_file = "dict.txt"
    embed_dict = load_word_embedding(word_file, embeding_file)

    en_bpe = "en-de.en.bpe"
    en_word_emb_file = "transformer.emb.en.txt"
    get_bpe_word_embedding(embed_dict, en_bpe, en_word_emb_file)

    de_bpe = "en-de.de.bpe"
    de_word_emb_file = "transformer.emb.de.txt"
    get_bpe_word_embedding(embed_dict, de_bpe, de_word_emb_file)