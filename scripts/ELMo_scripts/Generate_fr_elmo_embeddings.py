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

#input_text = []
#with open(os.path.join(data_dir, 'train.lang2'), 'r') as en_file:
#    for line in en_file.readlines():
#        line = line.rstrip().split('\n')
#        input_text.append(line[0]+" .")
#    en_file.close()

decoder_input_text = []
decoder_target_text = []

with open(os.path.join(data_dir, 'train.lang2'), 'r', encoding="UTF-8") as fr_file:
    for line in fr_file.readlines():
        line = line.rstrip().split('\n')

        target_line = line[0]+" </S>"+"\n"
        input_line = "<S> "+line[0]+"\n"

        decoder_input_text.append(input_line)
        decoder_target_text.append(target_line)
    fr_file.close()

print("text loaded : ", len(decoder_input_text))
print("text loaded : ", len(decoder_target_text))

tokenized_context_in = [sentence.split() for sentence in decoder_input_text]
tokenized_context_out = [sentence.split() for sentence in decoder_target_text]

max_seq_len_in = max([len(x) for x in tokenized_context_in])
max_seq_len_out = max([len(x) for x in tokenized_context_out])

print("max_seq_len_in = ", max_seq_len_in)
print("max_seq_len_out = ", max_seq_len_out)

# Location of pretrained LM.  Here we use the test fixtures.
datadir = "/project/cq-training-1/project2/teams/team08/ELMo/swb_fr"
vocab_file = os.path.join(datadir, 'vocab.txt')
options_file = os.path.join(datadir, 'options_eval.json')
weight_file = os.path.join(datadir, 'swb_weights_fr.hdf5')

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

elmo_emb_in = np.array([])
elmo_emb_out = np.array([])
dset_size = len(tokenized_context_in)

with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())

    # Create batches of data.
    for i in range(0, dset_size, batch_size):
        context_ids_in  = batcher.batch_sentences(tokenized_context_in[i:i+batch_size])
        context_ids_out = batcher.batch_sentences(tokenized_context_out[i:i+batch_size])
        #print("Shape of context ids = ", context_ids.shape)

        # Compute ELMo representations for input
        elmo_context_input_ = sess.run(
            elmo_context_input['weighted_op'],
            feed_dict={context_character_ids: context_ids_in}
        )
        
        # Compute ELMo representations for targets
        elmo_context_out_ = sess.run(
            elmo_context_input['weighted_op'],
            feed_dict={context_character_ids: context_ids_out}
        )

        #print("dset = ", dset.shape)
        #print("elmo_context_input_=",elmo_context_input_.shape)

        if i%1000==0:
            print("Batch done: ",i+1000)
            #print("elmo_context_input_=",elmo_context_input_.shape)

        #new_shape = (i+batch_size, max_seq_len, emb_dim)
        #dset.resize( new_shape )
        if i==0:
            elmo_emb_in  = pad_sequences(elmo_context_input_, maxlen=max_seq_len_in, padding='post', dtype='float32')
            elmo_emb_out = pad_sequences(elmo_context_out_  , maxlen=max_seq_len_out, padding='post', dtype='float32')
        else:
            elmo_emb_in  = np.vstack((elmo_emb_in , pad_sequences(elmo_context_input_, maxlen=max_seq_len_in, padding='post', dtype='float32')))
            elmo_emb_out = np.vstack((elmo_emb_out, pad_sequences(elmo_context_out_  , maxlen=max_seq_len_out, padding='post', dtype='float32')))
        
        #print("EMBD :", elmo_context_input_.shape)
        #print("EMBD :", elmo_context_input_[0])
        #elmo_emb.append(pad_sequences(elmo_context_input_, maxlen=max_seq_len, padding='pre', dtype='float32'))
        #print("PADDED :", elmo_emb.shape)

        #elmo_emb.append(elmo_context_input_)

#elmo_emb = np.array(dset)
print("Shape of generated input embeddings = ",elmo_emb_in.shape)
print("Shape of generated targets embeddings = ",elmo_emb_out.shape)

print("creating input embeddings file ...")
with h5py.File("ELMo_decoder_input_embeddings.hdf5", "w") as f:
    f.create_dataset("decoder", data=elmo_emb_in)
    f.close()
print("creating targets embeddings file ...")
with h5py.File("ELMo_decoder_output_embeddings.hdf5", "w") as f:
    f.create_dataset("decoder", data=elmo_emb_out)                                                                                                                                                                                                   
    f.close()
print("Done")

