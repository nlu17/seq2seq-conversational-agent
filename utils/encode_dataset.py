import sys
from vocabulary import Vocabulary

VOC_SIZE = 20000
CORPUS_SIZE = -1


"""
Get a file with sentences as input and writes to an output file the
sequences of tokens encoded using the token ids.
"""
def encode_dataset(vocab, data_file, out_file):
    with open(data_file, "r") as fin, open(out_file, "w") as fout:
        lines = fin.readlines()
        new_lines = [vocab.parse(line) for line in lines]
        new_lines = [vocab.get_tok_ids(line)
                     for line in filter(lambda x: x, new_lines)]
        for line in new_lines:
            fout.write(" ".join([str(x) for x in line]) + "\n")


if __name__ == "__main__":
    voc = Vocabulary()
    voc.load_from_file(sys.argv[1])

    # Create an alternative data set with tok_ids instead of tokens.
    encode_dataset(voc, sys.argv[2], sys.argv[3])
