import os
import io
import json
import matplotlib.pyplot as plt
from pretrained_tokenizer import Tokenizer
from transformer import *
import numpy as np


def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

def load_file(path):
    assert os.path.isfile(path), f"invalid config file: {path}"
    with open(path, "r") as fd:
        return json.load(fd)


def load_tokenizers(inp_language, target_language, user_config):
    # load pre-trained tokenizer
    pretrained_tokenizer_path_inp = user_config["tokenizer_path_{}".format(inp_language)]
    pretrained_tokenizer_path_tar = user_config["tokenizer_path_{}".format(target_language)]

    tokenizer_inp = Tokenizer(inp_language, pretrained_tokenizer_path_inp,
                              max_length=user_config["max_length_{}".format(inp_language)])
    tokenizer_tar = Tokenizer(target_language, pretrained_tokenizer_path_tar,
                              max_length=user_config["max_length_{}".format(target_language)])

    return tokenizer_inp, tokenizer_tar


def load_transformer_model(user_config, tokenizer_inp, tokenizer_tar):
    input_vocab_size = tokenizer_inp.vocab_size
    target_vocab_size = tokenizer_tar.vocab_size
    inp_language = user_config["inp_language"]
    target_language = user_config["target_language"]

    use_pretrained_emb = user_config["use_pretrained_emb"]
    if use_pretrained_emb:
        pretrained_weights_inp = np.load(user_config["pretrained_emb_path_{}".format(inp_language)])
        pretrained_weights_tar = np.load(user_config["pretrained_emb_path_{}".format(target_language)])
    else:
        pretrained_weights_inp = None
        pretrained_weights_tar = None

    # define custom learning scheduler
    learning_rate = CustomSchedule(user_config["transformer_model_dimensions"])
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    transformer_model = Transformer(user_config["transformer_num_layers"],
                                    user_config["transformer_model_dimensions"],
                                    user_config["transformer_num_heads"],
                                    user_config["transformer_dff"],
                                    input_vocab_size,
                                    target_vocab_size,
                                    en_input=input_vocab_size,
                                    fr_target=target_vocab_size,
                                    rate=user_config["transformer_dropout_rate"],
                                    weights_inp=pretrained_weights_inp,
                                    weights_tar=pretrained_weights_tar)

    ckpt = tf.train.Checkpoint(transformer=transformer_model,
                               optimizer=optimizer)

    checkpoint_path = user_config["transformer_checkpoint_path"]
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)
    
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored from path {}'.format(ckpt_manager.latest_checkpoint))

    return transformer_model, optimizer, ckpt_manager


def create_mix_dataset(synthetic_data_path_lang1, true_data_path_lang1, true_unaligned_data_path_lang2,
                       true_data_path_lang2, num_of_times_to_add_true_data: int):
    assert num_of_times_to_add_true_data > 0

    synthetic_data_lang1 = io.open(synthetic_data_path_lang1).read().strip().split('\n')
    true_aligned_data_lang1 = io.open(true_data_path_lang1).read().strip().split('\n')
    true_unaligned_data_lang2 = io.open(true_unaligned_data_path_lang2).read().strip().split('\n')
    true_aligned_data_lang2 = io.open(true_data_path_lang2).read().strip().split('\n')

    new_data_lang1, new_data_lang2 = synthetic_data_lang1, true_unaligned_data_lang2
    for i in range(num_of_times_to_add_true_data):
        new_data_lang1 += true_aligned_data_lang1
        new_data_lang2 += true_aligned_data_lang2

    shuffle_together = list(zip(new_data_lang1, new_data_lang2))
    np.random.shuffle(shuffle_together)
    new_data_lang1, new_data_lang2 = zip(*shuffle_together)

    return  new_data_lang1, new_data_lang2

def plot_attention_weights(attention_weights, sentence, result, layer):
    """Visualize layer attention in transformer model """
    fig = plt.figure(figsize=(16, 8))
    attention_weights = tf.squeeze(attention_weights[layer], axis=0)
    for head in range(attention_weights.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)
        # title and labels, setting initial sizes
        fig.suptitle('{}'.format(layer), fontsize=12)
        # plot the attention weights
        ax.matshow(attention_weights[head][:-1, :], cmap='viridis')
        fontdict = {'fontsize': 12}
        ax.set_xticks(range(len(sentence) + 2))
        ax.set_yticks(range(len(result)))
        ax.set_ylim(len(result) - 1.5, -0.5)
        ax.set_xticklabels(
            ['<start>'] + sentence + ['<end>'], fontdict=fontdict, rotation=90)
        ax.set_yticklabels(result, fontdict=fontdict)
        ax.set_xlabel('Head {}'.format(head + 1))
    plt.tight_layout()
    plt.show()
