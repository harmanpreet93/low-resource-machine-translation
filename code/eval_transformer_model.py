import os
import json
import argparse
from pretrained_tokenizer import Tokenizer
from data_loader import DataLoader
from transformer import *
from evaluator import compute_bleu
from tqdm import tqdm

"""Evaluate"""


def load_file(path):
    assert os.path.isfile(path), f"invalid config file: {path}"
    with open(path, "r") as fd:
        return json.load(fd)


def sacrebleu_metric(model, pred_file_path, target_file_path, tokenizer_fr, test_dataset, max_length):
    if target_file_path == None:
        with open(pred_file_path, "w", buffering=1) as f_pred:
            # evaluations possibly faster in batches
            for batch, (en_, fr_, fr) in enumerate(test_dataset):
                translated_batch = translate_batch(model, en_, tokenizer_fr, max_length)
                for i, pred in enumerate(translated_batch):
                    f_pred.write(pred.strip() + "\n")
    else:
        # write both prediction and target file together
        with open(pred_file_path, "w", buffering=1) as f_pred, open(target_file_path, "w", buffering=1) as f_true:
            for batch, (en_, fr_, fr) in enumerate(test_dataset):
                # evaluations possibly faster in batches
                translated_batch = translate_batch(model, en_, tokenizer_fr, max_length)
                for true, pred in zip(fr, translated_batch):
                    f_true.write(tf.compat.as_str_any(true.numpy()).strip() + "\n")
                    f_pred.write(pred.strip() + "\n")


def translate_batch(model, inp, tokenizer_fr, max_length):
    output, _ = evaluate_batch(model, inp, tokenizer_fr, max_length)
    predicted_sentences = []
    for pred in output.numpy():
        predicted_sentences.append(sequences_to_texts(tokenizer_fr, pred))
    return predicted_sentences


def evaluate_batch(model, inputs, tokenizer_fr, max_length):
    encoder_input = tf.convert_to_tensor(inputs)
    decoder_input = tf.expand_dims([tokenizer_fr.bos_token_id] * inputs.shape[0], axis=1)
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
        if (predicted_id == tokenizer_fr.eos_token_id).numpy().all():
            return tf.squeeze(output, axis=0), attention_weights

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


def do_evaluation(user_config, input_file_path, target_file_path, pred_file_path):
    # load pre-trained tokenizer
    pretrained_tokenizer_path_en = user_config["tokenizer_path_en"]
    tokenizer_en = Tokenizer('en', pretrained_tokenizer_path_en, max_length=user_config["max_length_en"])

    pretrained_tokenizer_path_fr = user_config["tokenizer_path_fr"]
    tokenizer_fr = Tokenizer('en', pretrained_tokenizer_path_fr, max_length=user_config["max_length_fr"])

    # data loader
    test_dataloader = DataLoader(user_config["transformer_batch_size"],
                                 input_file_path,
                                 target_file_path,
                                 tokenizer_en,
                                 tokenizer_fr,
                                 shuffle=False)
    test_dataset = test_dataloader.get_data_loader()

    learning_rate = CustomSchedule(user_config["transformer_model_dimensions"])
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    input_vocab_size = tokenizer_en.vocab_size
    target_vocab_size = tokenizer_fr.vocab_size

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
        print('Checkpoint restored!!')

    sacrebleu_metric(transformer_model,
                     pred_file_path,
                     target_file_path,
                     tokenizer_fr,
                     test_dataset,
                     tokenizer_fr.MAX_LENGTH
                     )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file containing training parameters", type=str, required=True)
    parser.add_argument("--input_file_path", help="File to generate translations for", type=str, required=True)
    parser.add_argument("--pred_file_path", help="Path to save predicted translations", type=str, required=True)
    parser.add_argument("--target_file_path",
                        help="Path to save true translations. If you already have true translations, don't pass anything. Else this will overwrite file.",
                        type=str, default=None)
    args = parser.parse_args()

    assert os.path.isfile(args.input_file_path), f"invalid input file: {args.input_file_path}"
    if args.target_file_path is not None:
        assert os.path.isfile(args.target_file_path), f"invalid target file: {args.target_file_path}"

    user_config = load_file(args.config)
    print(json.dumps(user_config, indent=2))

    # generate translations
    do_evaluation(user_config,
                  args.input_file_path,
                  args.target_file_path,
                  args.pred_file_path)

    if args.target_file_path is not None:
        print("Computing bleu score now...")
        # compute bleu score
        compute_bleu(args.input_file_path, args.target_file_path, print_all_scores=False)
    else:
        print("Not predicting bleu as --target_file_path was not provided")


if __name__ == "__main__":
    main()
