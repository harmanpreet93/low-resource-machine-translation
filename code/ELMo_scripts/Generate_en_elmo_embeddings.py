import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import scipy.spatial.distance as ds
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
import h5py
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Now we can compute embeddings.
data_dir = "/project/cq-training-1/project2/data"
batch_size = 100
#emb_dim = 1024

input_text = []
with open(os.path.join(data_dir, 'train.lang1'), 'r') as en_file:
    for line in en_file.readlines():
        line = line.rstrip().split('\n')
        input_text.append(line[0]+" .")
    en_file.close()

print("text loaded : ", len(input_text))

tokenized_context = [sentence.split() for sentence in input_text]
#print(tokenized_context)

max_seq_len = max([len(x) for x in tokenized_context])
print("max_seq_len = ", max_seq_len)

# Location of pretrained LM.  Here we use the test fixtures.
datadir = "/project/cq-training-1/project2/teams/team08/ELMo/swb_en2"
vocab_file = os.path.join(datadir, 'vocab.txt')
options_file = os.path.join(datadir, 'options_eval.json')
weight_file = os.path.join(datadir, 'swb_weights_en2.hdf5')

# Create a Batcher to map text to character ids.
batcher = Batcher(vocab_file, 50)

# Input placeholders to the biLM.
context_character_ids = tf.placeholder('int32', shape=(None, None, 50))

# Build the biLM graph.
bilm = BidirectionalLanguageModel(options_file, weight_file)

# Get ops to compute the LM embeddings.
context_embeddings_op = bilm(context_character_ids)

# Get an op to compute ELMo (weighted average of the internal biLM layers)
elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)

elmo_emb = np.array([])
dset_size = len(tokenized_context)

with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())

    # Create batches of data.
    for i in range(0, dset_size, batch_size):
        context_ids = batcher.batch_sentences(tokenized_context[i:i+batch_size])
        #print("Shape of context ids = ", context_ids.shape)

        # Compute ELMo representations (here for the input only, for simplicity).
        elmo_context_input_ = sess.run(
            elmo_context_input['weighted_op'],
            feed_dict={context_character_ids: context_ids}
        )
        #print("dset = ", dset.shape)
        #print("elmo_context_input_=",elmo_context_input_.shape)

        if i%1000==0:
            print("Batch done: ",i+1000)
            #print("elmo_context_input_=",elmo_context_input_.shape)

        #new_shape = (i+batch_size, max_seq_len, emb_dim)
        #dset.resize( new_shape )
        if i==0:
            elmo_emb = pad_sequences(elmo_context_input_, maxlen=max_seq_len, padding='post', dtype='float32')
        else:
            elmo_emb = np.vstack((elmo_emb,pad_sequences(elmo_context_input_, maxlen=max_seq_len, padding='post', dtype='float32')))
        
        #print("EMBD :", elmo_context_input_.shape)
        #print("EMBD :", elmo_context_input_[0])
        #elmo_emb.append(pad_sequences(elmo_context_input_, maxlen=max_seq_len, padding='pre', dtype='float32'))
        #print("PADDED :", elmo_emb.shape)

        #elmo_emb.append(elmo_context_input_)

#elmo_emb = np.array(dset)
print("Shape of generated embeddings = ",elmo_emb.shape)

print("creating embeddings file ...")
with h5py.File("ELMo_encoder_embeddings_post.hdf5", "w") as f:
    f.create_dataset("encoder", data=elmo_emb)
    f.close()
print("Done")

