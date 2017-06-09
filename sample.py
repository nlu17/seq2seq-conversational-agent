'''
Code in this file is for sampling use of chatbot
'''

from collections import Counter
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
bucket_curr_idxs = [0, 0, 0, 0]
max_source_length = 0
max_target_length = 0

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'data/checkpoints/', 'Directory to store/restore checkpoints')
flags.DEFINE_string('data_dir', "data/", "Data storage directory")
flags.DEFINE_string('with_attention', False, "If the model uses attention")
flags.DEFINE_string('ckpt_file', '', "Checkpoint file")
flags.DEFINE_string('output_file', '', 'Name of the file wo write outputs to')
flags.DEFINE_string('custom_decoder', "", "model decoder")

BATCH_SIZE = 256

if FLAGS.custom_decoder == "default":
    import tensorflow.contrib.legacy_seq2seq as seq2seq
elif FLAGS.custom_decoder == "mmi":
    import mmi_seq2seq as seq2seq
elif FLAGS.custom_decoder == "beam":
    import beam_seq2seq as seq2seq
else:
    raise NotImplementedError

def main():
    test_outputs = {}
    with tf.Session() as sess:
        vocab = vocab_utils.VocabMapper(FLAGS.data_dir)
        vocab_prior = vocab.getLogPrior()
        model = loadModel(sess, FLAGS.checkpoint_dir, FLAGS.ckpt_file, vocab_prior)

        test_set = readData(
                FLAGS.data_dir+"test_source.txt",
                FLAGS.data_dir+"test_target.txt")

        print(_buckets)
        model.batch_size = BATCH_SIZE
        vocab = vocab_utils.VocabMapper(FLAGS.data_dir)

        for bucket_id in range(len(_buckets)):
            batch = get_batch(test_set, bucket_id)
            while batch is not None:
                curr_batch_size, encoder_inputs, decoder_inputs, target_weights, input_ids = batch

                _, output_symbols, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                        target_weights, bucket_id, True)

                outputs = []
                if FLAGS.custom_decoder == "mmi":
                    outputs = [out for out in output_symbols]
                elif FLAGS.custom_decoder == "default":
                    # bucket_size x BATCH_SIZE x vocab_size
                    outputs = [np.argmax(logit, axis=1).astype(int) for logit in output_logits]
                else:
                    raise NotImplementedError

                if FLAGS.custom_decoder == "default":
                    probabilities = [softmax(logit) for logit in output_logits]
                    probabilities = [
                            probabilities[i][xrange(BATCH_SIZE), decoder_inputs[i]]
                            for i in xrange(len(decoder_inputs))]
                elif FLAGS.custom_decoder == "mmi":
                    probabilities = [softmax(logit) for logit in output_logits]
                    probabilities = [
                            probabilities[i][xrange(BATCH_SIZE), decoder_inputs[i]]
                            for i in xrange(len(decoder_inputs))]

                decoder_inputs = np.array(decoder_inputs).transpose()

                outputs = np.array(outputs).transpose()
                outputs = outputs[:curr_batch_size]
                probabilities = np.array(probabilities).transpose()
                probabilities = probabilities[:curr_batch_size]

                # Get first occurence of EOS.
                eos_idxs = np.argmax(outputs == vocab_utils.EOS_ID, axis=1)
                ref_eos_idxs = np.argmax(decoder_inputs == vocab_utils.EOS_ID, axis=1)

                mask = np.ones_like(probabilities)
                mask[:,0] = 0 # ignore prob of GO symbol
                for i, (out, eos, ref_eos, input_id) in enumerate(
                        zip(outputs, eos_idxs, ref_eos_idxs, input_ids)):
                    if out[eos] == vocab_utils.EOS_ID:
                        out = out[:eos]
                    if decoder_inputs[i][ref_eos] == vocab_utils.EOS_ID:
                        mask[i][ref_eos:] = 0
                    convo_output =  " ".join(vocab.indices2Tokens(out))

                    # Compute BLEU score.
                    intersection = Counter(out) & Counter(decoder_inputs[i])
                    bleu = (1 + sum(intersection.values())) / out.shape[0]

                    test_outputs[input_id] = [convo_output, bleu]

                # TODO: Compute perplexities
                log_probs = np.log2(probabilities) * mask
                perplexities = 2 ** (-np.sum(log_probs, axis=1) /
                        np.sum(mask, axis=1))

                for perp, input_id in zip(perplexities, input_ids):
                    test_outputs[input_id].append(perp)

                print("Finished batch with shape", outputs.shape)
                batch = get_batch(test_set, bucket_id)

    print("CACAT", test_outputs[0][0], test_outputs[0][1], test_outputs[0][2])
    with open(FLAGS.data_dir+FLAGS.output_file, "w") as fout:
        # Write to file in order.
        for input_id in xrange(len(test_outputs)):
            fout.write(test_outputs[input_id][0] + " " +
                    str(test_outputs[input_id][1]) + " " +
                    str(test_outputs[input_id][2]) + "\n")
        fout.flush()


def loadModel(session, path, checkpoint_file, vocab_prior=None):
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
        with_attention=FLAGS.with_attention,
        custom_decoder=FLAGS.custom_decoder, vocab_prior=vocab_prior)

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
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(vocab_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids, counter])
                        break
                else:
                    last_id = len(_buckets) - 1
                    source_size, target_size = _buckets[last_id]
                    source_ids = source_ids[:source_size]
                    target_ids = target_ids[:target_size]
                    data_set[last_id].append([source_ids, target_ids, counter])
                source, target = source_file.readline(), target_file.readline()
                counter += 1
    print(len(data_set[0]), len(data_set[1]), len(data_set[2]), len(data_set[3]))
    return data_set

def get_batch(data, bucket_id):
    encoder_size, decoder_size = _buckets[bucket_id]
    encoder_inputs, decoder_inputs, input_ids = [], [], []

#     print("CURR IDXS", bucket_curr_idxs)

    curr_size = 0
    for _ in xrange(BATCH_SIZE):
        idx = bucket_curr_idxs[bucket_id]

        if idx >= len(data[bucket_id]):
            break

        bucket_curr_idxs[bucket_id] += 1
        encoder_input, decoder_input, input_id = data[bucket_id][idx]

        encoder_pad = [vocab_utils.PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(encoder_input + encoder_pad)

        decoder_pad_size = decoder_size - len(decoder_input) - 1
        decoder_inputs.append([vocab_utils.GO_ID] + decoder_input +
                              [vocab_utils.PAD_ID] * decoder_pad_size)

        input_ids.append(input_id)
        curr_size += 1

    if encoder_inputs == []:
        return None

    # Add dummy entries to complete the batch.
    for _ in xrange(BATCH_SIZE - curr_size):
        encoder_inputs.append([vocab_utils.PAD_ID] * encoder_size)
        decoder_inputs.append([vocab_utils.PAD_ID] * decoder_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
        batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][length_idx]
                      for batch_idx in xrange(BATCH_SIZE)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
        batch_decoder_inputs.append(
            np.array([decoder_inputs[batch_idx][length_idx]
                      for batch_idx in xrange(BATCH_SIZE)], dtype=np.int32))

        # Create target_weights to be 0 for targets that are padding.
        batch_weight = np.ones(BATCH_SIZE, dtype=np.float32)
        for batch_idx in xrange(BATCH_SIZE):
            # We set weight to 0 if the corresponding target is a PAD symbol.
            # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == vocab_utils.PAD_ID:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)
    return curr_size, batch_encoder_inputs, batch_decoder_inputs, batch_weights, input_ids

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1).reshape((-1,1)))
    return e_x / e_x.sum(axis=1).reshape((-1,1))

if __name__=="__main__":
    main()
