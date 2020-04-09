import time
import tqdm
from transformer import *
from transformers import AutoTokenizer
from data_loader import DataLoader

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


"""Evaluate"""


def evaluate(model, inp_sentence, tokenizer_en, tokenizer_fr, max_length=40):
    # inp sentence is english, hence adding the start and end token
    inp_sentence = tokenizer_en.encode(inp_sentence)
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is french, the first word to the transformer
    # should be the french start token.
    decoder_input = [tokenizer_fr.bos_token]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = model(encoder_input,
                                               output,
                                               False,
                                               enc_padding_mask,
                                               combined_mask,
                                               dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer_fr.eos_token_id:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def main():
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    batch_size = 16
    EPOCHS = 5
    learning_rate = CustomSchedule(d_model)

    print("Loading tokenizers")
    # load pre-trained tokenizer
    # make sure the path contains files:
    # config.json, merges.txt, vocab.json, tokenizer_config.json, special_tokens_map.json
    # code to train new tokenizer is in train_custom_tokenizer.py
    pretrained_tokenizer_path_en = "../tokenizer_data_en"
    tokenizer_en = AutoTokenizer.from_pretrained(pretrained_tokenizer_path_en)

    pretrained_tokenizer_path_fr = "../tokenizer_data_fr"
    tokenizer_fr = AutoTokenizer.from_pretrained(pretrained_tokenizer_path_fr)

    input_vocab_size = tokenizer_en.vocab_size
    target_vocab_size = tokenizer_fr.vocab_size

    aligned_path_en = "../data/train.lang1"
    aligned_path_fr = "../data/train.lang2"
    train_dataloader = DataLoader(batch_size,
                                  aligned_path_en,
                                  aligned_path_fr,
                                  tokenizer_en,
                                  tokenizer_fr)

    train_dataset = train_dataloader.get_data_loader()

    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    transformer_model = Transformer(num_layers, d_model, num_heads, dff,
                                    input_vocab_size, target_vocab_size,
                                    pe_input=input_vocab_size,
                                    pe_target=target_vocab_size,
                                    rate=dropout_rate)

    checkpoint_path = "../log/checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer_model,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    with tqdm.tqdm("training", total=EPOCHS) as pbar:
        for epoch in range(EPOCHS):
            start = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()

            # inp -> english, tar -> french
            for (batch, (inp, tar)) in enumerate(train_dataset):
                train_step(transformer_model, loss_object, optimizer, inp, tar, train_loss, train_accuracy)

                if batch % 10 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, train_loss.result(), train_accuracy.result()))

            if (epoch + 1) % 2 == 0:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                train_loss.result(),
                                                                train_accuracy.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

            pbar.update(1)

if __name__ == "__main__":
    main()
