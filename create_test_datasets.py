# given a file with triplets generate test_source.txt and test_target.txt that will be used later 
import util.tokenizer
import os
import sys
import util.vocabutils as vocab_utils
from multiprocessing import Process


DATA_DIR = "data/"
MAX_NUM_LINES = 2  # The maximum number of lines for conversational history.

class DataProcessor(object):

	def __init__(self, max_target_length=100, max_source_length=100, output_test_source = "test_source.txt", output_test_target = "test_target.txt"):

		self.MAX_SOURCE_TOKEN_LENGTH = max_source_length
		self.MAX_TARGET_TOKEN_LENGTH = max_target_length
		self.tokenizer = util.tokenizer.basic_tokenizer

		self.test_source_file = os.path.join(DATA_DIR, output_test_source)
		self.test_target_file = os.path.join(DATA_DIR, output_test_target)

		print("Checking to see what data processor needs to do...")

		self.vocab_path = os.path.join(DATA_DIR, "vocab.pkl")

	def run(self, test_text_file):

        #create vocab file
		if not os.path.isfile(self.vocab_path):
			vocab_builder = vocab_utils.VocabBuilder(self.max_vocab_size, DATA_DIR)
			print("Building vocab...")
			#loop through data
			for text_file in train_text_files:
				with open(text_file, "r+") as f:
					vocab_builder.growVocab(f.read())
				print("Creating vocab file...")
			vocab_builder.createVocabFile()

        
		self.vocab_mapper = vocab_utils.VocabMapper(DATA_DIR)
		#create source and target token id files
        
		print("Creating token id data source and target test files...")
		print("This is going to take a while...")
		

		self.parseTextFile(text_file = test_text_file)

	def parseTextFile(self, text_file):

		print('Text file: ', text_file)
		with open(text_file, "r+") as f:
			convos = f.readlines()
			for convo in convos:
				convo = convo.strip().split("\t")

				line_buffer = []
				for line in convo:
					line_buffer.append(line)
					if len(line_buffer) > MAX_NUM_LINES or \
						len(line_buffer) == len(convo):
						self.findSentencePairs(line_buffer)
						line_buffer.pop(0)

	def findSentencePairs(self, convo):

			#check whether any of the triples has a length > 80
			for one_sample in convo:
				sentence = one_sample.strip()
				words = self.tokenizer(sentence)

				if len(words) >= self.MAX_SOURCE_TOKEN_LENGTH or len(words) >= self.MAX_TARGET_TOKEN_LENGTH:
					return



			for i in range(1, len(convo)):
	            # TODO: Use first two utterances as source
				#             source_sentences = " ".join(convo[:i])
				source_sentences = convo[i-1].strip()
				target_sentence = convo[i].strip()
	            #Tokenize sentences
				source_sentences = self.tokenizer(source_sentences)
				target_sentence = self.tokenizer(target_sentence)

	            #Convert tokens to id string, reverse source inputs
				source_sentences = list(reversed(self.vocab_mapper.tokens2Indices(source_sentences)))
				target_sentence = self.vocab_mapper.tokens2Indices(target_sentence)
				#remove outliers (really long sentences) from data
				if len(source_sentences) >= self.MAX_SOURCE_TOKEN_LENGTH or \
					len(target_sentence) >= self.MAX_TARGET_TOKEN_LENGTH:
					#print("skipped {0} and {1}".format(len(source_sentences), len(target_sentence)))
					continue
				source_sentences = " ".join([str(x) for x in source_sentences])
				target_sentence = " ".join([str(x) for x in target_sentence])

				data_source = self.test_source_file
				data_target = self.test_target_file

				with open(data_source, "a+") as f2:
					f2.write(source_sentences + "\n")
				with open(data_target, "a+") as f2:
					f2.write(target_sentence + "\n")

test_path = sys.argv[1]

print('Test path: ', test_path)

processor = DataProcessor(max_target_length=80, max_source_length=80, output_test_source = "test_source.txt", output_test_target = "test_target.txt")
processor.run(test_path)
#processor.parseTextFile('data/Validation_Shuffled_Dataset.txt')