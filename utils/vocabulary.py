import numpy as np
import operator
import tensorflow as tf


class TokInfo:
    def __init__(self, tok, idx, one_hot, collection_count):
        self.tok = tok
        self.idx = idx
        self.collection_count = collection_count
        self.one_hot = one_hot

    def __str__(self):
        return self.tok + " " + str(self.one_hot) + " " + \
            str(self.collection_count)


class Vocabulary:
    def __init__(self, voc_size=20000, corpus_size=-1, max_sentence_len=30):
        self.voc_size = voc_size
        self.corpus_size = corpus_size
        self.max_sentence_len = max_sentence_len
        self.voc = {}  # token -> TokInfo
        self.sorted_voc = []  # list of (token, collection_count) tuples

    def init(self, corpus_file):
        tmp_dict = {}
        with open(corpus_file, "r") as fin:
            lines = fin.readlines()
            for line in lines[:self.corpus_size]:
                tokens = line.strip('\n').split(' ')
                for tok in tokens:
                    tmp_dict[tok] = tmp_dict.get(tok, 0) + 1
            tmp_dict.update({
                "<bos>": len(lines),
                "<eos>": len(lines),
                "<pad>": len(lines),
                "<unk>": len(lines)
            })

        self.sorted_voc = sorted(
                tmp_dict.items(),
                key=operator.itemgetter(1),
                reverse=True)
        self.sorted_voc = self.sorted_voc[:self.voc_size]

        encodings = self.create_one_hot_encodings()

        self.voc = dict(
            [(item[0], TokInfo(item[0], index, encodings[index], item[1]))
                for index, item in enumerate(self.sorted_voc)])

    """
    Get a sentence (string) as input and return a list of tokens. Returns None
    if the sentence size is above the threshold.
    """
    def parse(self, sentence):
        tokens = sentence.split(' ')
        if len(tokens) > self.max_sentence_len-2:
            return None

        # Check if there are out-of-vocabulary tokens.
        tokens = [tok if tok in self.voc else "<unk>" for tok in tokens]

        # Add bos/eos symbols.
        tokens = ["<bos>"] + tokens + ["<eos>"]

        # Pad the sentence.
        tokens = tokens + ["<pad>"] * (self.max_sentence_len - len(tokens))

        return tokens

    """
    Get a list of tokens and return a numpy array of indexes.
    """
    def get_tok_ids(self, tokens):
        return np.array([self.voc[tok].idx for tok in tokens])

    def load_from_file(self, voc_file):
        with open(voc_file, "r") as fin:
            lines = fin.readlines()

            encodings = self.create_one_hot_encodings()

            def cast(l):
                return (l[0], int(l[2]))
            self.sorted_voc = \
                [cast(line.strip("\n").split(' ')) for line in lines]
            self.voc = dict(
                [(item[0], TokInfo(item[0], index, encodings[index], item[1]))
                    for index, item in enumerate(self.sorted_voc)])

    """
    Write sorted_vec to file. The tokens are ordered by the collection count.
    """
    def dump_to_file(self, out_file):
        with open(out_file, "w") as fout:
            for index, item in enumerate(self.sorted_voc[:self.voc_size]):
                fout.write("{0} {1} {2}\n".format(item[0], index, item[1]))

    def create_one_hot_encodings(self):
        # Create one hot encodings for words.
        encodings = None
        with tf.Session() as sess:
            encodings = sess.run(
                tf.one_hot(np.arange(self.voc_size), self.voc_size))
        return encodings
