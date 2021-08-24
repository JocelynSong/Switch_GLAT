import sys
import torch

from fairseq.data import Dictionary


dict_path = "./data/dict.txt"
all_dict = Dictionary.load(dict_path)


def save_data(path, dictionary, save_path):
    f = open(path, 'r', encoding='utf-8')

    lines = f.readlines()
    all_tokens = []
    for line in lines:
        tokens = dictionary.encode_line_not_to_tensor(
            line, add_if_not_exist=False,
            append_eos=True, reverse_order=False,
        )
        all_tokens.append(tokens)

    data = {"sentences": all_tokens}
    torch.save(data, save_path, pickle_protocol=4)


if __name__ == "__main__":
    source_file = sys.argv[1]
    target_file = sys.argv[2]
    save_data(source_file, all_dict, target_file)



