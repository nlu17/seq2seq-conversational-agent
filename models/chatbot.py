'''
This is adapted from tensorflow translation example:
https://github.com/tensorflow/tensorflow/blob/e39d8feebb9666a331345cd8d960f5ade4652bba/tensorflow/models/rnn/translate/seq2seq_model.py
'''
import random

import numpy as np
from six.moves import xrange
import tensorflow as tf
import mmi_functions as mmi

import util.vocabutils as vocab_utils

class ChatbotModel(object):
    def __init__(self, vocab_size, buckets, hidden_size, dropout,
                 num_layers, max_gradient_norm, batch_size, learning_rate,
                 lr_decay_factor, num_samples=512, forward_only=False,
                 with_attention=False, custom_decoder=False, vocab_prior=None):
        '''
        vocab_size: number of vocab tokens
        buckets: buckets of max sequence lengths
        hidden_size: dimension of hidden layers
        num_layers: number of hidden layers
        max_gradient_norm: maximum gradient magnitude
        batch_size: number of training examples fed to network at once
        learning_rate: starting learning rate of network
        lr_decay_factor: amount by which to decay learning rate
        num_samples: number of samples for sampled softmax
        forward_only: Whether to build backpass nodes or not
        with_attention: attention model or not
        custom_decoder: MMI Decoder or not
        vocab_prior: useful for MMI Decoder
        '''
        self.vocab_size = vocab_size
        self.buckets = buckets
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * lr_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.dropout_keep_prob_lstm_input = tf.constant(self.dropout)
        self.dropout_keep_prob_lstm_output = tf.constant(self.dropout)
        self.with_attention = with_attention
        self.custom_decoder = custom_decoder
        self.vocab_prior = vocab_prior

        output_projection = None
        softmax_loss_function = None
        if num_samples > self.vocab_size:
            num_samples = self.vocab_size - 1
        if num_samples > 0 and num_samples < self.vocab_size:
            w = tf.get_variable("proj_w", [hidden_size, self.vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [self.vocab_size])
            output_projection = (w, b)

        #def sampled_loss(labels, inputs):
        #    labels = tf.reshape(labels, [-1, 1])
        #    return tf.nn.sampled_softmax_loss(w_t, b, labels, inputs, num_samples,
        #                                        self.vocab_size)
        #softmax_loss_function = sampled_loss

        #cell = tf.contrib.rnn.DropoutWrapper(
        #    tf.contrib.rnn.BasicLSTMCell(hidden_size),
        #    input_keep_prob=self.dropout_keep_prob_lstm_input,
        #    output_keep_prob=self.dropout_keep_prob_lstm_output)
        #if num_layers > 1:
        #    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)

        def sampled_loss(labels, inputs):
            labels = tf.reshape(labels, [-1, 1])
            # We need to compute the sampled_softmax_loss using 32bit floats to
            # avoid numerical instabilities.
            local_w_t = tf.cast(w_t, tf.float32)
            local_b = tf.cast(b, tf.float32)
            local_inputs = tf.cast(inputs, tf.float32)
            return tf.cast(
                        tf.nn.sampled_softmax_loss(
                            weights=local_w_t,
                            biases=local_b,
                            labels=labels,
                            inputs=local_inputs,
                            num_sampled=num_samples,
                            num_classes=self.vocab_size),
                            dtype=tf.float32)
        softmax_loss_function = sampled_loss

        def single_cell():
            return tf.contrib.rnn.DropoutWrapper( \
                tf.contrib.rnn.BasicLSTMCell(hidden_size), \
                input_keep_prob=self.dropout_keep_prob_lstm_input, \
                output_keep_prob=self.dropout_keep_prob_lstm_output)

        cell = single_cell()
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])

        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode_with_mmi, with_attention):
            s2s = None
            if do_decode_with_mmi:
                tf.contrib.legacy_seq2seq.embedding_rnn_decoder = mmi.get_embedding_mmi_rnn_decoder(self.vocab_prior)
                tf.contrib.legacy_seq2seq.embedding_attention_decoder = mmi.get_embedding_mmi_attention_decoder(self.vocab_prior)
            if with_attention:
                s2s = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, cell, num_encoder_symbols=vocab_size,
                    num_decoder_symbols=vocab_size, embedding_size=hidden_size,
                    output_projection=output_projection, feed_previous=do_decode_with_mmi)
            else:
                s2s = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                    encoder_inputs, decoder_inputs, cell, num_encoder_symbols=vocab_size,
                    num_decoder_symbols=vocab_size, embedding_size=hidden_size,
                    output_projection=output_projection, feed_previous=do_decode_with_mmi)

            return s2s

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                        name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                        name="weight{0}".format(i)))

        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]

        if forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True, self.with_attention),
                softmax_loss_function=softmax_loss_function)
            if output_projection is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, False, self.with_attention),
                softmax_loss_function=softmax_loss_function)

        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            with tf.name_scope("train"):
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(
                    gradients,
                    max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))
        self.saver = tf.train.Saver(tf.global_variables())

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

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
        '''
        Inputs:

        Returns:

        '''
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None
        else:
            return None, outputs[0], outputs[1:]
