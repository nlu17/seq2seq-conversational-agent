'''
Code in this file is for sampling use of chatbot
'''


import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import sys
import os
from six.moves import xrange
import models.chatbot
import util.hyperparamutils as hyper_params
import util.vocabutils as vocab_utils
from os import listdir
from os.path import isfile, join

_buckets = []
max_source_length = 0
max_target_length = 0

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'data/checkpoints/', 'Directory to store/restore checkpoints')
flags.DEFINE_string('data_dir', "data/", "Data storage directory")
flags.DEFINE_string('with_attention', False, "If the model uses attention")
flags.DEFINE_string('ckpt_file', '', "Checkpoint file")
flags.DEFINE_string('output_file', '', 'Name of the file wo write outputs to')

BATCH_SIZE = 128

# TODO: make sure you write the outputs in order

def main():
    test_set = readData(source_test_file_path, target_test_file_path)

    with tf.Session() as sess, open(FLAGS.data_dir+FLAGS.output_file, "w") as fout:
        model = loadModel(sess, FLAGS.checkpoint_dir, FLAGS.ckpt_file)
        print(_buckets)
        model.batch_size = 1
        vocab = vocab_utils.VocabMapper(FLAGS.data_dir)
        batch = sentences[BATCH_SIZE]
        curr_idx = BATCH_SIZE
        conversation_history = [sentence]

        while curr_idx < len(sentences):
            token_ids = list(reversed(vocab.tokens2Indices(" ".join(conversation_history))))
            eligible_bucket_ids = [b for b in xrange(len(_buckets))
                    if _buckets[b][0] > len(token_ids)]

            if len(eligible_bucket_ids) > 0:
                bucket_id = min(eligible_bucket_ids)

                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)

                _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                        target_weights, bucket_id, True)

                #TODO implement beam search
                outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

                if vocab_utils.EOS_ID in outputs:
                    outputs = outputs[:outputs.index(vocab_utils.EOS_ID)]

                convo_output =  " ".join(vocab.indices2Tokens(outputs))

                conversation_history.append(convo_output)
                fout.write(sentence + "\t" + convo_output + "\n")
                fout.flush()

            sentence = sentences[curr_idx]
            curr_idx += 1
            conversation_history.append(sentence)
            conversation_history = conversation_history[-1:]


def loadModel(session, path, checkpoint_file):
    global _buckets
    global max_source_length
    global max_target_length
    params = hyper_params.restoreHyperParams(path)
    buckets = []
    num_buckets = params["num_buckets"]
    max_source_length = params["max_source_length"]
    max_target_length = params["max_target_length"]
    for i in range(num_buckets):
        buckets.append((params["bucket_{0}_target".format(i)],
            params["bucket_{0}_target".format(i)]))
        _buckets = buckets
    model = models.chatbot.ChatbotModel(params["vocab_size"], _buckets,
        params["hidden_size"], 1.0, params["num_layers"], params["grad_clip"],
        1, params["learning_rate"], params["lr_decay_factor"], 512, True,
        with_attention=FLAGS.with_attention)

    print("Reading model parameters from {0}".format(checkpoint_file))
    model.saver.restore(session, checkpoint_file)
    return model


def readData(source_path, target_path):
    '''
    This method directly from tensorflow translation example
    '''
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target:
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(vocab_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set

# TODO: change this to be deterministic, not random
def get_batch(self, data, bucket_id):
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    for _ in xrange(self.batch_size):
        encoder_input, decoder_input = random.choice(data[bucket_id])

        encoder_pad = [vocab_utils.PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(encoder_input + encoder_pad)

        decoder_pad_size = decoder_size - len(decoder_input) - 1
        decoder_inputs.append([vocab_utils.GO_ID] + decoder_input +
                              [vocab_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
        batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][length_idx]
                      for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
        batch_decoder_inputs.append(
            np.array([decoder_inputs[batch_idx][length_idx]
                      for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        # Create target_weights to be 0 for targets that are padding.
        batch_weight = np.ones(self.batch_size, dtype=np.float32)
        for batch_idx in xrange(self.batch_size):
            # We set weight to 0 if the corresponding target is a PAD symbol.
            # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == vocab_utils.PAD_ID:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

if __name__=="__main__":
    main()
