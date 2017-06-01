import util.dataprocessor as data_utils
import util.vocabutils as vocabutils

DATA_DIR = "data/"

print("Train/Test files not detected, creating now...")
data_processor = data_utils.DataProcessor(
    max_vocab_size=10000,
    tokenizer_str="basic",
    max_source_length=80,
    max_target_length=80)
data_processor.run()

with open(DATA_DIR + "train_source.txt") as f:
    lines = f.readlines()
    print(lines[:5])
    lines = [[int(x) for x in line.strip().split()] for line in lines]

vocab_mapper = vocabutils.VocabMapper(DATA_DIR)
lines = list(map(lambda line: " ".join(vocab_mapper.indices2Tokens(line)), lines))
print(lines[:5])
