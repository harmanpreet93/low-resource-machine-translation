from train_transformer_model import *
from evaluator import *

"""Evaluate"""


def evaluate_batch(model, inputs, tokenizer_en, tokenizer_fr, max_length=40):
    # # inp sentence is english, hence adding the start and end token
    # # inp_sentence = tokenizer_en.encode(inp_sentence)
    # encoder_input = tf.expand_dims(inp_sentence, 0)
    # # as the target is french, the first word to the transformer
    # # should be the french start token.
    # decoder_input = [tokenizer_fr.bos_token_id]
    # output = tf.expand_dims(decoder_input, 0)

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

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return output, attention_weights


def sequences_to_texts(tokenizer, pred):
    decoded_text = tokenizer.decode(pred).split(tokenizer.eos_token)[0]
    decoded_text_ = tokenizer.decode(tokenizer.encode(decoded_text), skip_special_tokens=True)
    return decoded_text_


def translate_batch(model, inp, tokenizer_en, tokenizer_fr, max_length=512):
    output, _ = evaluate_batch(model, inp, tokenizer_en, tokenizer_fr, max_length)
    predicted_sentences = []
    for pred in output.numpy():
        predicted_sentences.append(sequences_to_texts(tokenizer_fr, pred))
    return predicted_sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file containing training parameters", type=str)
    args = parser.parse_args()
    user_config = load_file(args.config)

    pretrained_tokenizer_path_en = user_config["tokenizer_path_en"]
    tokenizer_en = AutoTokenizer.from_pretrained(pretrained_tokenizer_path_en)

    pretrained_tokenizer_path_fr = user_config["tokenizer_path_en"]
    tokenizer_fr = AutoTokenizer.from_pretrained(pretrained_tokenizer_path_fr)

    # data loader
    test_aligned_path_en = user_config["test_english_data_path"]
    test_aligned_path_fr = user_config["test_french_data_path"]
    test_dataloader = DataLoader(user_config["transformer_batch_size"] // 4,
                                 test_aligned_path_en,
                                 test_aligned_path_fr,
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

    input_file_path = "../log/predicted_fr_1.txt"
    target_file_path = "../log/true_fr_1.txt"
    with open(input_file_path, "w", buffering=1) as f_pred, open(target_file_path, "w", buffering=1) as f_true:
        for batch, (en_, fr_, fr) in enumerate(test_dataset):
            translated_batch = translate_batch(transformer_model, en_, tokenizer_en, tokenizer_fr, max_length=300)
            for true, pred in zip(fr, translated_batch):
                f_true.write(tf.compat.as_str_any(true.numpy()).strip() + "\n")
                f_pred.write(pred.strip() + "\n")

    # compute bleu score
    if user_config["compute_bleu"]:
        compute_bleu(input_file_path, target_file_path, print_all_scores=False)


if __name__ == "__main__":
    main()
