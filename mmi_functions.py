from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
import tensorflow as tf

GAMMA = 3    # word limit for anti language penalty
LAMBDA = 0.5  # weight of anti-language compare to p(T|S)

def _extract_mmiargmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True,
                              log_prior=None):
  """Get a loop_function that extracts the previous symbol and embeds it.
  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.
  Returns:
    A loop function.
  """

  def loop_function(prev, i):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])

    if i < GAMMA:
      log_probts = tf.nn.log_softmax(prev)      # p(T|S)
      log_probts_sub = tf.subtract(log_probts, tf.scalar_mul(LAMBDA, log_prior))   # p(T|S) - λ.p(T)
      print("here")
      prev_symbol = math_ops.argmax(log_probts_sub, 1)
    else:
      prev_symbol = math_ops.argmax(prev, 1)

    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev

  return loop_function

def get_embedding_mmi_rnn_decoder(log_prior):

    def embedding_mmi_rnn_decoder(decoder_inputs,
                                  initial_state,
                                  cell,
                                  num_symbols,
                                  embedding_size,
                                  output_projection=None,
                                  feed_previous=False,
                                  update_embedding_for_previous=True,
                                  scope=None):
          """RNN decoder with embedding and a pure-decoding option.
          """
          with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
              if output_projection is not None:
                dtype = scope.dtype
                proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
                proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
                proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
                proj_biases.get_shape().assert_is_compatible_with([num_symbols])

              embedding = variable_scope.get_variable("embedding",
                                                      [num_symbols, embedding_size])
              loop_function = _extract_mmiargmax_and_embed(
                  embedding, output_projection,
                  update_embedding_for_previous, log_prior) if feed_previous else None
              emb_inp = (embedding_ops.embedding_lookup(embedding, i)
                      for i in decoder_inputs)
              return rtf.contrib.legacy_seq2seq.nn_decoder(
                  emb_inp, initial_state, cell, loop_function=loop_function)

    return embedding_mmi_rnn_decoder

def get_embedding_mmi_attention_decoder(log_prior):

    def embedding_mmi_attention_decoder(decoder_inputs,
                                        initial_state,
                                        attention_states,
                                        cell,
                                        num_symbols,
                                        embedding_size,
                                        num_heads=1,
                                        output_size=None,
                                        output_projection=None,
                                        feed_previous=False,
                                        update_embedding_for_previous=True,
                                        dtype=None,
                                        scope=None,
                                        initial_state_attention=False):
        """RNN decoder with embedding and attention and a pure-decoding option.
        """
        if output_size is None:
            output_size = cell.output_size
        if output_projection is not None:
            proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
            proj_biases.get_shape().assert_is_compatible_with([num_symbols])

        with variable_scope.variable_scope(
            scope or "embedding_attention_decoder", dtype=dtype) as scope:

            embedding = variable_scope.get_variable("embedding",
                                                    [num_symbols, embedding_size])
            loop_function = _extract_mmiargmax_and_embed(
                embedding, output_projection,
                update_embedding_for_previous, log_prior) if feed_previous else None
            emb_inp = [
                embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs
            ]
            return tf.contrib.legacy_seq2seq.attention_decoder(
                emb_inp,
                initial_state,
                attention_states,
                cell,
                output_size=output_size,
                num_heads=num_heads,
                loop_function=loop_function,
                initial_state_attention=initial_state_attention)
    
    return embedding_mmi_attention_decoder
