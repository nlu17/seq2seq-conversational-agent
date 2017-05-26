import gensim
import numpy as np
import tensorflow as tf


def load_embedding(session, vocab, emb, path, dim_embedding):
    '''
      session        Tensorflow session object
      vocab          A Vocabulary instance
      emb            Embedding tensor of shape vocabulary_size x dim_embedding
      path           Path to embedding file
      dim_embedding  Dimensionality of the external embedding.
    '''
    print("Loading external embeddings from %s" % path)
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
    external_embedding = np.zeros(shape=(vocab.voc_size, dim_embedding))
    matches = 0
    for tok, tok_info in vocab.voc.items():
        # Remove angular brackets for the special symbols (e.g. <bos>, <unk>,
        # <pad>, <eos>).
        tok = tok.strip("<>") if len(tok) >= 2 else tok
        if tok in model.vocab:
            external_embedding[tok_info.idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[tok_info.idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)

    print("%d words out of %d could be loaded" % (matches, vocab.voc_size))

    pretrained_embeddings = tf.placeholder(tf.float32, [None, None])
    assign_op = emb.assign(pretrained_embeddings)
    session.run(assign_op, {pretrained_embeddings: external_embedding})
