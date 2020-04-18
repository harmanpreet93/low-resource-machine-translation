import argparse
import time
from data_loader import DataLoader
from evaluator import compute_bleu
from utils import *

"""Evaluate"""


def sacrebleu_metric(model, pred_file_path, target_file_path, tokenizer_tar, test_dataset, max_length):
    start = time.time()
    if target_file_path is None:
        with open(pred_file_path, "w", buffering=1) as f_pred:
            # evaluations possibly faster in batches
            for batch, (inp_seq, tar_seq, tar) in enumerate(test_dataset):
                if (batch+1) % 4 == 0:
                    print("Evaluating batch {}".format(batch))
                translated_batch = translate_batch(model, inp_seq, tokenizer_tar, max_length)
                for i, pred in enumerate(translated_batch):
                    f_pred.write(pred.strip() + "\n")
    else:
        # write both prediction and target file together
        with open(pred_file_path, "w", buffering=1) as f_pred, open(target_file_path, "w", buffering=1) as f_true:
            for batch, (inp_seq, tar_seq, tar) in enumerate(test_dataset):
                # evaluations possibly faster in batches
                translated_batch = translate_batch(model, inp_seq, tokenizer_tar, max_length)
                for true, pred in zip(tar, translated_batch):
                    f_true.write(tf.compat.as_str_any(true.numpy()).strip() + "\n")
                    f_pred.write(pred.strip() + "\n")
    print('Time taken to compute sacrebleu files: {} secs'.format(time.time() - start))


def translate_batch(model, inp, tokenizer_tar, max_length):
    output, _ = evaluate_batch(model, inp, tokenizer_tar, max_length)
    predicted_sentences = []
    for pred in output.numpy():
        predicted_sentences.append(sequences_to_texts(tokenizer_tar, pred))
    return predicted_sentences


def evaluate_batch(model, inputs, tokenizer_tar, max_length):
    encoder_input = tf.convert_to_tensor(inputs)
    decoder_input = tf.expand_dims([tokenizer_tar.bos_token_id] * inputs.shape[0], axis=1)
    output = decoder_input
    attention_weights = None

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
        if (predicted_id == tokenizer_tar.eos_token_id).numpy().all():
            return output, attention_weights
            # return tf.squeeze(output, axis=0), attention_weights

        # concatenate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return output, attention_weights


def sequences_to_texts(tokenizer, pred):
    # split because batch might flush output tokens even after eos token due to race conditions
    pad_indices = tf.where(pred == tokenizer.eos_token_id)
    if pad_indices.shape[0] > 0:
        index = pad_indices.numpy()[0][0]
        pred = pred[:index]
    decoded_text = tokenizer.decode(pred)
    return decoded_text


# Since the target sequences are padded, it is important
# to apply a padding mask when calculating the loss.
def loss_function(real, pred, loss_object, pad_token_id):
    """Calculates total loss containing cross entropy with padding ignored.
      Args:
        logits: Tensor of size [batch_size, length_logits, vocab_size]
        labels: Tensor of size [batch_size, length_labels]
        loss_object: Cross entropy loss
        pad_token_id: Pad token id to ignore
      Returns:
        A scalar float tensor for loss.
    """
    mask = tf.math.logical_not(tf.math.equal(real, pad_token_id))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def train_step(model, loss_object, optimizer, inp, tar,
               train_loss, train_accuracy, pad_token_id):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions, _ = model(inp, tar_inp,
                               True,
                               enc_padding_mask,
                               combined_mask,
                               dec_padding_mask)
        loss = loss_function(tar_real, predictions, loss_object, pad_token_id)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


def do_evaluation(user_config, input_file_path, target_file_path, pred_file_path):
    inp_language = user_config["inp_language"]
    target_language = user_config["target_language"]

    print("\n****Evaluating model from {} to {}****\n".format(inp_language, target_language))

    print("****Loading Sub-Word Tokenizers****")
    # load pre-trained tokenizer
    tokenizer_inp, tokenizer_tar = load_tokenizers(inp_language, target_language, user_config)

    print("****Loading DataLoader****")
    # data loader
    dummy_dataloader = DataLoader(user_config["transformer_batch_size"] * 2,
                                  user_config["dummy_data_path_{}".format(inp_language)],
                                  None,
                                  tokenizer_inp,
                                  tokenizer_tar,
                                  inp_language,
                                  target_language,
                                  False)
    dummy_dataset = dummy_dataloader.get_data_loader()

    # data loader
    test_dataloader = DataLoader(user_config["transformer_batch_size"] * 2,
                                 input_file_path,
                                 target_file_path,
                                 tokenizer_inp,
                                 tokenizer_tar,
                                 inp_language,
                                 target_language,
                                 False)
    test_dataset = test_dataloader.get_data_loader()

    input_vocab_size = tokenizer_inp.vocab_size
    target_vocab_size = tokenizer_tar.vocab_size

    use_pretrained_emb = user_config["use_pretrained_emb"]
    if use_pretrained_emb:
        pretrained_weights_inp = np.load(user_config["pretrained_emb_path_{}".format(inp_language)])
        pretrained_weights_tar = np.load(user_config["pretrained_emb_path_{}".format(target_language)])
    else:
        pretrained_weights_inp = None
        pretrained_weights_tar = None

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

    # print("****Generating Translations after, without load weights****")
    sacrebleu_metric(transformer_model,
                     pred_file_path,
                     None,
                     tokenizer_tar,
                     dummy_dataset,
                     tokenizer_tar.MAX_LENGTH
                     )

    # load model
    model_path = user_config["model_file"]
    transformer_model.load_weights(model_path)

    print("****Generating Translations after, with load weights****")
    sacrebleu_metric(transformer_model,
                     pred_file_path,
                     target_file_path,
                     tokenizer_tar,
                     test_dataset,
                     tokenizer_tar.MAX_LENGTH
                     )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file containing training parameters", type=str, required=True)
    parser.add_argument("--input_file_path", help="File to generate translations for", type=str, required=True)
    parser.add_argument("--pred_file_path", help="Path to save predicted translations", type=str, required=True)
    parser.add_argument("--target_file_path",
                        help="Path to save true translations. If you already have true translations, "
                             "don't pass anything. Else this will overwrite file.",
                        type=str, default=None)
    args = parser.parse_args()

    assert os.path.isfile(args.input_file_path), f"invalid input file: {args.input_file_path}"
    if args.target_file_path is not None:
        assert os.path.isfile(args.target_file_path), f"invalid target file: {args.target_file_path}"

    user_config = load_file(args.config)
    print(json.dumps(user_config, indent=2))
    seed = user_config["random_seed"]
    set_seed(seed)

    # generate translations
    do_evaluation(user_config,
                  args.input_file_path,
                  None,
                  args.pred_file_path)

    if args.target_file_path is not None:
        print("\nComputing bleu score now...")
        # compute bleu score
        compute_bleu(args.pred_file_path, args.target_file_path, print_all_scores=False)
    else:
        print("\nNot predicting bleu as --target_file_path was not provided")


if __name__ == "__main__":
    main()
