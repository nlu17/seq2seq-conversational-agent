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
import mmi_functions as mmi
from os import listdir
from os.path import isfile, join

_buckets = []
max_source_length = 0
max_target_length = 0
#_buckets = [(10, 10), (50, 15), (100, 20), (200, 50)]

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'data/checkpoints/', 'Directory to store/restore checkpoints')
flags.DEFINE_string('data_dir', "data/", "Data storage directory")
flags.DEFINE_boolean('with_attention', False, "If the model uses attention")
flags.DEFINE_boolean('custom_decoder', False, "If the model uses mmi decoder")
flags.DEFINE_string('ckpt_file', '', "Checkpoint file")
flags.DEFINE_integer('static_temp', 60, 'number between 0 and 100. The lower the number the less likely static responses will come up')
#flags.DEFINE_string('text', 'Hello World!', 'Text to sample with.')


def main():
        with tf.Session() as sess:
                vocab = vocab_utils.VocabMapper(FLAGS.data_dir)
                vocab_prior = vocab.getLogPrior()
                model = loadModel(sess, FLAGS.checkpoint_dir, FLAGS.ckpt_file, vocab_prior)
                print(_buckets)
                model.batch_size = 1
                sys.stdout.write(">")
                sys.stdout.flush()
                sentence = sys.stdin.readline().lower()
                conversation_history = [sentence]

                while sentence:
                        token_ids = list(reversed(vocab.tokens2Indices(" ".join(conversation_history))))
                        bucket_id = min([b for b in xrange(len(_buckets))
                                if _buckets[b][0] > len(token_ids)])

                        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        {bucket_id: [(token_ids, [])]}, bucket_id)

                        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                target_weights, bucket_id, True)

                        # outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                        outputs = []
                        for i,logit in enumerate(output_logits):
                          if i < mmi.GAMMA:
                            log_probts = tf.nn.log_softmax(logit)      # p(T|S)
                            log_probts_sub = tf.subtract(log_probts, tf.scalar_mul(mmi.LAMBDA, vocab_prior))   # p(T|S) - λ.p(T)
                            output = tf.argmax(log_probts_sub, 1)
                          else:
                            output = tf.argmax(logit, 1)
                          outputs.append(sess.run(output)[0])
                          #print(outputs[-1])

                        if vocab_utils.EOS_ID in outputs:
                                outputs = outputs[:outputs.index(vocab_utils.EOS_ID)]

                        convo_output =  " ".join(vocab.indices2Tokens(outputs))

                        conversation_history.append(convo_output)
                        print(convo_output)
                        sys.stdout.write(">")
                        sys.stdout.flush()
                        sentence = sys.stdin.readline().lower()
                        conversation_history.append(sentence)
                        conversation_history = conversation_history[-1:]

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
        1, params["learning_rate"], params["lr_decay_factor"], num_samples=512, forward_only=True,
        with_attention=FLAGS.with_attention, custom_decoder=FLAGS.custom_decoder, vocab_prior=vocab_prior)

    print("Reading model parameters from {0}".format(checkpoint_file))
    model.saver.restore(session, checkpoint_file)
    return model


if __name__=="__main__":
    main()
