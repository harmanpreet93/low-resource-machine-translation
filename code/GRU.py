import time
import utils_GRU
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Encoder with GRU gates
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        # using pre-trained embeddings
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # we are doing this to broadcast addition along the time axis to
        # calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(
            tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


# GRU decoder
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        # Here we specify the pre-trained embeddings
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fc = tf.keras.layers.Dense(vocab_size)
        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none")
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def train_step(inp, targ, enc_hidden, targ_lang, batch_sz, encoder, decoder,
               optimizer):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index["<start>"]] *
                                   batch_sz, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden,
                                                 enc_output)
            loss += loss_function(targ[:, t], predictions)
            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = loss / int(targ.shape[1])
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


def do_training(GRU_config):
    # mapping english sentence to french sentence on the same line in a new file
    utils_GRU.combine_files(
        GRU_config["aligned_en_path"],
        GRU_config["aligned_fr_path"],
        GRU_config["data_folder"] + "en-fr.txt",
    )

    input_tensor, target_tensor, inp_lang, targ_lang = utils_GRU.load_dataset(
        GRU_config["data_folder"] + "en-fr.txt")

    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
        input_tensor,
        target_tensor,
        test_size=0.1,
        random_state=GRU_config["random_seed"])

    input_df, target_df = pd.DataFrame(input_tensor), pd.DataFrame(
        target_tensor)
    input_df_train, input_df_val, target_df_train, target_df_val = train_test_split(
        input_df,
        target_df,
        test_size=0.1,
        random_state=GRU_config["random_seed"])

    indices_en_val = list(input_df_val.index)
    indices_fr_val = list(target_df_val.index)

    # writes out english and french test sentences to files
    utils_GRU.map_indices(GRU_config, "test_en.txt", indices_en_val, "en")
    utils_GRU.map_indices(GRU_config, "test_fr.txt", indices_fr_val, "fr")

    optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    buffer_size = len(input_tensor_train)
    steps_per_epoch = len(input_tensor_train) // GRU_config["batch_size"]
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1
    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_train, target_tensor_train)).shuffle(buffer_size)
    dataset = dataset.batch(GRU_config["batch_size"], drop_remainder=True)
    encoder = Encoder(
        vocab_inp_size,
        GRU_config["embedding_dim"],
        GRU_config["hidden_units"],
        GRU_config["batch_size"],
    )
    attention_layer = BahdanauAttention(10)
    decoder = Decoder(
        vocab_tar_size,
        GRU_config["embedding_dim"],
        GRU_config["hidden_units"],
        GRU_config["batch_size"],
    )

    # defining checkpoints
    ckpt = tf.train.Checkpoint(optimizer=optimizer,
                               encode=encoder,
                               decoder=decoder)
    checkpoint_path = GRU_config["GRU_checkpoint_path"]
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              checkpoint_path,
                                              max_to_keep=10)

    # if a checkpoint exists, restore the latest
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Lastest checkpoint restored!")

    epochs = GRU_config["nb_epochs"]
    for epoch in range(epochs):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(
                inp,
                targ,
                enc_hidden,
                targ_lang,
                GRU_config["batch_size"],
                encoder,
                decoder,
                optimizer,
            )
            total_loss += batch_loss
            if batch % 100 == 0:
                print("Epoch {} Batch {} Loss {:.4f}".format(
                    epoch + 1, batch, batch_loss.numpy()))

        if (epoch + 1) % 2 == 0:
            ckpt_save_path = ckpt_manager.save()
            print("Saving checkpoint for epoch {} at {}".format(
                epoch + 1, ckpt_save_path))

        print("Epoch {} Loss {:.4f}".format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print("Time taken for 1 epoch {} sec\n".format(time.time() - start))

    # save model parameters after last epoch
    ckpt_save_path = ckpt_manager.save()
    print("Training finished. Saving checkpoint for {} epoch at {}".format(
        epochs, ckpt_save_path))
    return optimizer, encoder, decoder


def generate_predictions(GRU_config):
    """
    Function that generates predictions to file
    """
    optimizer, encoder, decoder = do_training(GRU_config)

    checkpoint_path = GRU_config["GRU_checkpoint_path"]
    ckpt = tf.train.Checkpoint(optimizer=optimizer,
                               encoder=encoder,
                               decoder=decoder)
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              checkpoint_path,
                                              max_to_keep=10)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Checkpoint restored!")

    input_tensor, target_tensor, inp_lang, targ_lang = utils_GRU.load_dataset(
        GRU_config["data_folder"] + "en-fr.txt")

    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1

    max_length_targ, max_length_inp = (
        utils_GRU.max_length(target_tensor),
        utils_GRU.max_length(input_tensor),
    )

    with open(GRU_config["val_data_path_en"], "r", encoding="UTF-8") as f:
        # writes out predictions in a file called predictions.txt
        with open(GRU_config["data_folder"] + "predictions.txt",
                  "w",
                  encoding="UTF-8") as outfile:
            for index, sentence in enumerate(f):
                sent = utils_GRU.preprocess_sentence(sentence)
                # if word not in vocab output random word index
                inputs = [
                    inp_lang.word_index[i]
                    if i in inp_lang.word_index else np.random.randint(
                        low=0, high=len(inp_lang.word_index), size=1)
                    for i in sent.split(" ")
                ]

                inputs = tf.keras.preprocessing.sequence.pad_sequences(
                    [inputs], maxlen=max_length_inp, padding="post")
                inputs = tf.convert_to_tensor(inputs)
                result = ""
                hidden = [tf.zeros((1, GRU_config["hidden_units"]))]
                enc_out, enc_hidden = encoder(inputs, hidden)
                dec_hidden = enc_hidden
                dec_input = tf.expand_dims([targ_lang.word_index["<start>"]],
                                           0)
                sentence_list = []

                for t in range(max_length_targ):
                    predictions, dec_hidden, attention_weights = decoder(
                        dec_input, dec_hidden, enc_out)
                    predicted_id = tf.argmax(predictions[0]).numpy()
                    result = targ_lang.index_word[predicted_id]
                    sentence_list.append(result)
                    if targ_lang.index_word[predicted_id] == "<end>":
                        break
                    # the predicted ID is fed back into the model
                    dec_input = tf.expand_dims([predicted_id], 0)

                if "<end>" in sentence_list:
                    sentence_list.remove("<end>")
                outfile.write(" ".join([str(word)
                                        for word in sentence_list]) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Configuration file containing training parameters",
        type=str)
    args = parser.parse_args()
    GRU_config = utils_GRU.load_file(args.config)
    generate_predictions(GRU_config)


if __name__ == "__main__":
    main()
