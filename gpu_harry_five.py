


import json
import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import pandas as pd
import argparse
from utility_gpt import *
import model, sample, encoder


if not files_exist():
    generate_look_ups()


def interact_model(
    model_name='124M',       
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
    constant = 0.0,
    counter = 0, 
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = str(os.path.dirname(os.path.abspath(__file__))) + '/models_gpt'
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    length = 30 #100
    word_set = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/word_sets_num.npy')
    numbers = word_set[counter]
    evaluations = np.zeros((8))
    if constant>0:
        text_all = open(str(os.path.dirname(os.path.abspath(__file__))) +"/results/snippets_with_anchoring.txt", "a+")
    else:
        text_all = open(str(os.path.dirname(os.path.abspath(__file__))) +"/results/snippets_no_anchoring.txt", "a+")

    text_all.write('==================================================================================================')
    text_all.write('\n')
    file1 = open(str(os.path.dirname(os.path.abspath(__file__))) + "/look_ups_gpt-2/word_sets.txt","r+")  
    line = file1.readlines()
    text_all.write(line[counter])
    text_all.write('\n')
    text_all.write('\n')

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        glovers = load_words_and_glove()
        converter_table = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/converter_table.npy')
        container = related_words()
        np.random.seed(seed)
        tf.set_random_seed(seed)
        weight = constant
        output, probabilites = sample.sample_sequence_glove_all_top_five_gpu(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p, glove= [glovers[numbers[0],:], glovers[numbers[1],:], glovers[numbers[2],:],  glovers[numbers[3],:], glovers[numbers[4],:]], #[glovers[0,:], glovers[98,:], glovers[2,:],  glovers[19,:], glovers[85,:]] #converter_table[14836,:] #words[98,:] #converter_table[5536,:]  #glover
            weight=weight
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)
        holder = 0
        counterer = 0
        perplexities = np.zeros((10))
        Nr_words = np.zeros((10))
        Nr_related = np.zeros((10))
        Nr_related_with = np.zeros((10))
        text_length = np.zeros((10))
        Nr_main = np.zeros((10))
        Nr_main_related_without = np.zeros((10))
        Nr_main_related_with = np.zeros((10))
        while holder<10:
            Harry, counterer = Harry_sentences_no_capital(counterer, 2)
            context_tokens = enc.encode(Harry)
            generated = 0
            for _ in range(nsamples // batch_size):
                out, proba = sess.run([output, probabilites], feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })
                out = out[:, len(context_tokens):]
                for i in range(batch_size):

                    main = [[numbers[0]]]
                    counter_main_sim_without = isSubset(container[numbers[0]][1:],out[i])
                    counter_main_sim_with = isSubset(container[numbers[0]][0:],out[i])
                    counter_main = counter_main_sim_with-counter_main_sim_without
                    checker = tokens_from_words(numbers)
                    counter_tot= 0
                    counter_sim = 0
                    counter_sim_with = 0
                    cond = False
                    counter = isSubset(checker,out[i])
                    counter_tot+= counter
                    if counter > 0:
                        cond = True
                    for num in numbers:
                        counter_sim += isSubset(container[num][1:],out[i])
                        counter_sim_with += isSubset(container[num][0:],out[i])

                    perplexitiy = np.power(proba,(-1/length))
                    generated += 1
                    text = enc.decode(out[i])
                    text_all.write(text)
                    text_all.write('\n')
                    perplexities[holder] = perplexitiy
                    Nr_words[holder] = counter_tot
                    Nr_related[holder] = counter_sim
                    text_length[holder] = text.count(' ')
                    Nr_main[holder] = counter_main
                    Nr_main_related_with[holder] = counter_main_sim_with
                    Nr_main_related_without[holder] = counter_main_sim_without
                    Nr_related_with[holder] =  counter_sim_with

            holder+=1

    evaluations[0] = np.mean(perplexities)
    evaluations[1] = np.mean(Nr_main)
    evaluations[2] = np.mean(Nr_main_related_with)
    evaluations[3] = np.mean(Nr_main_related_without)
    evaluations[4] = np.mean(Nr_words)
    evaluations[5] = np.mean(Nr_related_with)
    evaluations[6] = np.mean(Nr_related)
    evaluations[7] = np.mean(text_length)
    text_all.write('==================================================================================================')
    text_all.close()
    return evaluations

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-constant', type = float, default = 0.0)
    args = parser.parse_args()
    evaluations = np.zeros((40,8))
    constant = args.constant
    file1 = open(str(os.path.dirname(os.path.abspath(__file__))) + '/look_ups_gpt-2/word_sets.txt',"r+")  
    length = len(file1.readlines())
    for i in range(length):
        evaluations[i,:] = interact_model(counter=i, constant=constant)
    
    if constant>0:
        np.save(file=str(os.path.dirname(os.path.abspath(__file__))) + '/results/evaluations_with_anchoring', arr=evaluations) 
    else:
        np.save(file=str(os.path.dirname(os.path.abspath(__file__))) + '/results/evaluations_no_anchoring', arr=evaluations) 
    

