import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import model
import os

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, _ = logits.shape.as_list()
    sorted_logits = tf.contrib.framework.sort(logits, direction='DESCENDING', axis=-1)        #replaced tf.sort with tf.contrib.framework.sort because of issuesin version 1.12.0
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack([
        tf.range(0, batch),
        # number of indices to include
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )





def sample_sequence_glove_all_top_five_gpu(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=1, glove=None, weight=None):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)
    
    converter_table = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/converter_table.npy')

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        def body(past, prev, output, probabilities):
            next_outputs = step(hparams, prev, past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, p=top_p)

            glove_one, glove_two, glove_three, glove_four, glove_five = glove

            similar_five = tf.convert_to_tensor(cosine_similarity(np.reshape(glove_five,(1,-1)),converter_table),dtype=tf.float32)
            similar_four = tf.convert_to_tensor(cosine_similarity(np.reshape(glove_four,(1,-1)),converter_table),dtype=tf.float32)
            similar_three = tf.convert_to_tensor(cosine_similarity(np.reshape(glove_three,(1,-1)),converter_table),dtype=tf.float32)
            similar_two = tf.convert_to_tensor(cosine_similarity(np.reshape(glove_two,(1,-1)),converter_table),dtype=tf.float32)
            similar_one = tf.convert_to_tensor(cosine_similarity(np.reshape(glove_one,(1,-1)),converter_table),dtype=tf.float32)

            value = weight #7.0 #6.0  #8.0
            fact = tf.constant(value,tf.float32)

            prob = tf.nn.softmax(logits)


            logits = tf.add(logits, similar_one*fact)
            logits = tf.add(logits, similar_two*fact)
            logits = tf.add(logits, similar_three*fact)
            logits = tf.add(logits, similar_four*fact)
            logits = tf.add(logits, similar_five*fact)

            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            sample = samples[0,0]
            
            probability_old = tf.cast(tf.gather_nd(prob,[[0,sample]]),tf.float64)
            probability = tf.multiply(probability_old, probabilities)
            return [
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                samples,
                tf.concat([output, samples], axis=1),
                probability
            ]
        probabilities = tf.ones([1],tf.float64)
        past, prev, output, probabilities = body(None, context, context, probabilities)


        def cond(*args):
            return True

        _, _, tokens, probabilities = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=[
                past,
                prev,
                output,
                probabilities
            
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size,]),
                
            ],
            back_prop=False,
        )

        return tokens, probabilities


