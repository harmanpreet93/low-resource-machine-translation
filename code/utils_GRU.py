import re
import os
import json
import numpy as np
import tensorflow as tf


def preprocess_sentence(w):
    w = w.strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "é", "è", "ê", "ç", "î", "ï", "ô", "à", "û")
    w = re.sub(r"[^a-zA-Z?.!,¿éèçêîôïàû]+", " ", w)
    w = w.strip()
    # adding a start and an end token to the sentence
    w = '<start> ' + w + ' <end>'
    return w


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(
        tensor, padding='post')
    return tensor, lang_tokenizer


def combine_files(path_to_en_file, path_to_fr_file, output_path):
    """
    Creates a 1-to-1 mapping of english sentences to french sentences
    """
    with open(path_to_en_file, 'r', encoding='UTF-8') as en:
        with open(path_to_fr_file, 'r', encoding='UTF-8') as fr:
            with open(output_path, 'w', encoding='UTF-8') as outfile:
                en_lines = en.readlines()
                fr_lines = fr.readlines()
                for i in range(len(en_lines)):
                    line = en_lines[i].strip() + '\t' + fr_lines[i]
                    outfile.write(line)


def create_dataset(path, num_examples):
    """ 
    Cleans the sentences and return  
    """
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split(
        '\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    inp_lang, targ_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def load_file(path):
    assert os.path.isfile(path), f"invalid config file: {path}"
    with open(path, "r") as fd:
        return json.load(fd)


def target_to_file(input_path, output_path, indices):
    """ 
    Function that writes only the sentences used in the train-test split to a file 
    so it can be used to write targets.txt for evaluator.py or the input to write_prediction_file
    """
    with open(input_path, 'r') as f:
        with open(output_path, 'w') as outfile:
            for index, line in enumerate(f):
                if index in indices:
                    line = line.strip() + "\n"
                    outfile.write(line)


def create_embedding_matrix(inp_lang, embedding_dim, embedding_model):
    print('Preparing embedding matrix')
    oov = 0
    oov_words = []
    vocab_size = len(inp_lang.word_index) + 1
    word_index = inp_lang.word_index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if word not in embedding_model:
            # we can investigate the oov words
            oov_words.append(word)
            oov += 1
            # words not found in embedding_model will all be set to 0
            embedding_vector = np.zeros((embedding_dim,))
        else:
            embedding_vector = embedding_model.get_vector(word)
        embedding_matrix[i] = embedding_vector
    print(
        f"Found {oov} out-of-vocabulary words (not present in the embedding model)")
    return embedding_matrix


def map_indices(user_config, suffix, indices, lang):
    """ 
    Function that write sentences used in either training or validation set to file
    """
    data_path = user_config["data_folder"]
    if lang == 'en':
        input_file = user_config["aligned_en_path"]
    elif lang == 'fr':
        input_file = user_config["aligned_fr_path"]
    else:
        raise Exception('language not recognized')

    with open(input_file, 'r', encoding='UTF-8') as f:
        with open(data_path + suffix, 'w', encoding='UTF-8') as outfile:
            for index, line in enumerate(f):
                if index in indices:
                    line = line.strip() + "\n"
                    outfile.write(line)
