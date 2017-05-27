'''
Process the raw text data to create the following:

1. vocabulary file
2. source_train_file, target_train_file (index mapped train set files)
3. source_test_file, target_test_file (index mapped test set files)

TODO:
Some very minor parallelization takes place where train and test sets are
created in parallel. A much better parallelization can be achieved. It takes too
much time to process the data currently.

Clean up the code duplication
'''

import os
import util.tokenizer
import util.vocabutils as vocab_utils
from multiprocessing import Process

DATA_DIR = "data/"


class DataProcessor(object):
    def __init__(self, max_vocab_size, tokenizer_str, train_frac=0.5,
            max_target_length=500, max_source_length=500):
        '''
        Inputs:
        max_vocab_size: max size of vocab allowed
        tokenizer_str: string, type of tokenizer to use
        max_target_length: max length of target sentence
        max_source_length: max length of source sentence
        '''
        self.MAX_SOURCE_TOKEN_LENGTH = max_source_length
        self.MAX_TARGET_TOKEN_LENGTH = max_target_length
        self.tokenizer = util.tokenizer.basic_tokenizer
        assert train_frac > 0.0 and train_frac <= 1.0, "Train frac not between 0 and 1..."
        self.train_frac = train_frac
        self.max_vocab_size = max_vocab_size

        self.train_source_file = os.path.join(DATA_DIR, "train_source.txt")
        self.train_target_file = os.path.join(DATA_DIR, "train_target.txt")
        self.test_source_file = os.path.join(DATA_DIR, "test_source.txt")
        self.test_target_file = os.path.join(DATA_DIR, "test_target.txt")

        print("Checking to see what data processor needs to do...")
        self.vocab_path = os.path.join(DATA_DIR, "vocab.txt")

    def run(self):
        if not os.path.exists(self.train_source_file) \
                or not os.path.exists(self.train_target_file) \
                or not os.path.exists(self.test_source_file) \
                or not os.path.exists(self.test_target_file):
            print("Obtaining raw text conversation files...")
            train_text_files, test_text_files = self.getRawFileList()

        #create vocab file
        if not os.path.isfile(self.vocab_path):
            vocab_builder = vocab_utils.VocabBuilder(
                self.max_vocab_size,
                DATA_DIR)
            print("Building vocab...")
            #loop through data
            for text_file in train_text_files:
                with open(text_file, "r+") as f:
                    vocab_builder.growVocab(f.read())
            print("Creating vocab file...")
            vocab_builder.createVocabFile()

        if not os.path.exists(self.train_source_file) \
                or not os.path.exists(self.train_target_file) \
                or not os.path.exists(self.test_source_file) \
                or not os.path.exists(self.test_target_file):
            self.vocab_mapper = vocab_utils.VocabMapper(DATA_DIR)
            #create source and target token id files
            processes = []
            print("Creating token id data source and target train files...")

            p1 = Process(target=self.loopParseTextFiles, args=([train_text_files], True))
            p1.start()
            processes.append(p1)
            print("Creating token id data source and target test files...")
            print("This is going to take a while...")
            p2 = Process(target=self.loopParseTextFiles, args=([test_text_files], False))
            p2.start()
            processes.append(p2)

            for p in processes:
                if p.is_alive():
                    p.join()
            print("Done data pre-processing...")

    def loopParseTextFiles(self, text_files, is_train):
        # TODO: wtf?! why is it text_files[0] here?
        for text_file in text_files[0]:
            self.parseTextFile(text_file, is_train)

    def parseTextFile(self, text_file, is_train):
        with open(text_file, "r+") as f:
            lines = f.readlines()
            for line in lines:
                triple = line.strip().split("\t")
                self.findSentencePairs(triple, is_train)

    def getRawFileList(self):
        train = [os.path.join(DATA_DIR, "Training_Shuffled_Dataset.txt")]
        test = [os.path.join(DATA_DIR, "Validation_Shuffled_Dataset.txt")]
        return train, test

    def findSentencePairs(self, triple, is_train):
        for i in range(1, len(triple)):
            # TODO: maybe use first two utterances as source
#             source_sentences = " ".join(triple[:i])
            source_sentence = triple[i-1].strip()
            target_sentence = triple[i].strip()
            #Tokenize sentences
            source_sentence = self.tokenizer(source_sentence)
            target_sentence = self.tokenizer(target_sentence)

            #Convert tokens to id string, reverse source inputs
            source_sentence = list(reversed(self.vocab_mapper.tokens2Indices(source_sentence)))
            target_sentence = self.vocab_mapper.tokens2Indices(target_sentence)
            #remove outliers (really long sentences) from data
            if len(source_sentence) >= self.MAX_SOURCE_TOKEN_LENGTH or \
                len(target_sentence) >= self.MAX_TARGET_TOKEN_LENGTH:
                print("skipped {0} and {1}".format(len(source_sentence), len(target_sentence)))
                continue
            source_sentence = " ".join([str(x) for x in source_sentence])
            target_sentence = " ".join([str(x) for x in target_sentence])

            data_source = self.train_source_file
            data_target = self.train_target_file
            if not is_train:
                data_source = self.test_source_file
                data_target = self.test_target_file

            with open(data_source, "a+") as f2:
                f2.write(source_sentence + "\n")
            with open(data_target, "a+") as f2:
                f2.write(target_sentence + "\n")