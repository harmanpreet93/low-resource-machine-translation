import time
from transformer import *
from transformers import AutoTokenizer
from data_loader import DataLoader
import os
import json
import argparse

"""
### Masking

Mask all the pad tokens in the batch of sequence. 
It ensures that the model does not treat padding as the input. 
The mask indicates where pad value 0 is present: it outputs a 1 at those locations, and a 0 otherwise.
"""


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


"""
The look-ahead mask is used to mask the future tokens in a sequence. 
In other words,  the mask indicates which entries should not be used. 
"""


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(loss_object, real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.
#
# train_step_signature = [
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
# ]


def train_step(model, loss_function, optimizer, inp, tar, train_loss, train_accuracy):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = model(inp, tar_inp,
                               True,
                               enc_padding_mask,
                               combined_mask,
                               dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


def val_step(model, loss_function, inp, tar, val_loss, val_accuracy):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    predictions, _ = model(inp, tar_inp,
                           True,
                           enc_padding_mask,
                           combined_mask,
                           dec_padding_mask)
    loss = loss_function(tar_real, predictions)

    val_loss(loss)
    val_accuracy(tar_real, predictions)


def load_file(path):
    assert os.path.isfile(path), f"invalid config file: {path}"
    with open(path, "r") as fd:
        return json.load(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file containing training parameters", type=str)
    args = parser.parse_args()
    user_config = load_file(args.config)

    # load pre-trained tokenizer
    # make sure the path contains files:
    # config.json, merges.txt, vocab.json, tokenizer_config.json, special_tokens_map.json
    # code to train new tokenizer is in train_custom_tokenizer.py
    pretrained_tokenizer_path_en = user_config["tokenizer_path_en"]
    tokenizer_en = AutoTokenizer.from_pretrained(pretrained_tokenizer_path_en)

    pretrained_tokenizer_path_fr = user_config["tokenizer_path_en"]
    tokenizer_fr = AutoTokenizer.from_pretrained(pretrained_tokenizer_path_fr)

    input_vocab_size = tokenizer_en.vocab_size
    target_vocab_size = tokenizer_fr.vocab_size

    # data loader
    train_aligned_path_en = user_config["train_english_data_path"]
    train_aligned_path_fr = user_config["train_french_data_path"]
    train_dataloader = DataLoader(user_config["transformer_batch_size"],
                                  train_aligned_path_en,
                                  train_aligned_path_fr,
                                  tokenizer_en,
                                  tokenizer_fr)
    train_dataset = train_dataloader.get_data_loader()

    val_aligned_path_en = user_config["val_english_data_path"]
    val_aligned_path_fr = user_config["val_french_data_path"]
    val_dataloader = DataLoader(user_config["transformer_batch_size"] * 2, # for fast validation
                                val_aligned_path_en,
                                val_aligned_path_fr,
                                tokenizer_en,
                                tokenizer_fr)
    val_dataset = val_dataloader.get_data_loader()

    learning_rate = CustomSchedule(user_config["transformer_model_dimensions"])
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    transformer_model = Transformer(user_config["transformer_num_layers"],
                                    user_config["transformer_model_dimensions"],
                                    user_config["transformer_num_heads"],
                                    user_config["transformer_dff"],
                                    input_vocab_size,
                                    target_vocab_size,
                                    en_input=input_vocab_size,
                                    fr_target=target_vocab_size,
                                    rate=user_config["transformer_dropout_rate"])

    ckpt = tf.train.Checkpoint(transformer=transformer_model,
                               optimizer=optimizer)

    checkpoint_path = user_config["transformer_checkpoint_path"]
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    print("\nTraining model now...")
    for epoch in range(user_config["transformer_epochs"]):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> english, tar -> french
        for (batch, (inp, tar, _)) in enumerate(train_dataset):
            train_step(transformer_model, loss_object, optimizer, inp, tar, train_loss, train_accuracy)

            if batch % 50 == 0:
                print('Train: Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        print('Time taken for training {} epoch: {} secs\n'.format(epoch + 1, time.time() - start))

        # evaluate model every y-epochs
        if epoch % 2 == 0:
            start = time.time()
            print("\nRunning validation now...")
            val_loss.reset_states()
            val_accuracy.reset_states()
            # inp -> english, tar -> french
            for (batch, (inp, tar, _)) in enumerate(val_dataset):
                val_step(transformer_model, loss_object, inp, tar, val_loss, val_accuracy)

            print('Time taken for validation {} epoch: {} secs\n'.format(epoch + 1, time.time() - start))

            print('Val: Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                     val_loss.result(),
                                                                     val_accuracy.result()))

            # save model every y-epochs
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

        print('Train: Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                   train_loss.result(),
                                                                   train_accuracy.result()))

    # save model after last epoch
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))


if __name__ == "__main__":
    main()