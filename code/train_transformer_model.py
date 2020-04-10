import time
from transformer import *
from transformers import AutoTokenizer
from data_loader import DataLoader
import argparse
from eval_transformer_model import load_file, sacrebleu_metric


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
                           False,
                           enc_padding_mask,
                           combined_mask,
                           dec_padding_mask)
    loss = loss_function(tar_real, predictions)

    val_loss(loss)
    val_accuracy(tar_real, predictions)



def do_training(user_config):
    # load pre-trained tokenizer
    # make sure the path contains files:
    # config.json, merges.txt, vocab.json, tokenizer_config.json, special_tokens_map.json
    # code to train new tokenizer is in train_custom_tokenizer.py
    pretrained_tokenizer_path_en = user_config["tokenizer_path_en"]
    tokenizer_en = AutoTokenizer.from_pretrained(pretrained_tokenizer_path_en)

    pretrained_tokenizer_path_fr = user_config["tokenizer_path_fr"]
    tokenizer_fr = AutoTokenizer.from_pretrained(pretrained_tokenizer_path_fr)

    input_vocab_size = tokenizer_en.vocab_size
    target_vocab_size = tokenizer_fr.vocab_size

    # data loader
    train_aligned_path_en = user_config["train_data_path_en"]
    train_aligned_path_fr = user_config["train_data_path_fr"]
    train_dataloader = DataLoader(user_config["transformer_batch_size"],
                                  train_aligned_path_en,
                                  train_aligned_path_fr,
                                  tokenizer_en,
                                  tokenizer_fr)
    train_dataset = train_dataloader.get_data_loader()

    val_aligned_path_en = user_config["val_data_path_en"]
    val_aligned_path_fr = user_config["val_data_path_fr"]
    val_dataloader = DataLoader(user_config["transformer_batch_size"] * 2,  # for fast validation
                                val_aligned_path_en,
                                val_aligned_path_fr,
                                tokenizer_en,
                                tokenizer_fr)
    val_dataset = val_dataloader.get_data_loader()

    test_aligned_path_en = user_config["test_data_path_en"]
    test_aligned_path_fr = user_config["test_data_path_fr"]
    test_dataloader = DataLoader(user_config["transformer_batch_size"],
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
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    epochs = user_config["transformer_epochs"]
    print("\nTraining model now...")
    for epoch in range(epochs):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> english, tar -> french
        for (batch, (inp, tar, _)) in enumerate(train_dataset):
            train_step(transformer_model, loss_object, optimizer, inp, tar, train_loss, train_accuracy)

            if batch % 50 == 0:
                print('Train: Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        print('Time taken for training {} epoch: {} secs'.format(epoch + 1, time.time() - start))

        # evaluate and save model every y-epochs
        if epoch % 2 == 0:
            start = time.time()
            print("\nRunning validation now...")
            val_loss.reset_states()
            val_accuracy.reset_states()
            # inp -> english, tar -> french
            for (batch, (inp, tar, _)) in enumerate(val_dataset):
                val_step(transformer_model, loss_object, inp, tar, val_loss, val_accuracy)

            print('Val: Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                     val_loss.result(),
                                                                     val_accuracy.result()))

            print('Time taken for validation {} epoch: {} secs'.format(epoch + 1, time.time() - start))

            # save model every y-epochs
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

        if user_config["compute_bleu"]:
            # TODO: change manual paths
            if epoch % 200 == 0:
                print("\nComputing BLEU while training: ")
                input_file_path = "../log/predicted_fr_1.txt"
                target_file_path = "../log/true_fr_1.txt"
                sacrebleu_metric(transformer_model,
                                 input_file_path,
                                 target_file_path,
                                 tokenizer_en,
                                 tokenizer_fr,
                                 test_dataset,
                                 process_batches=False
                                 )

        print('Train: Epoch {} Loss {:.4f} Accuracy {:.4f}\n'.format(epoch + 1,
                                                                   train_loss.result(),
                                                                   train_accuracy.result()))

    # save model after last epoch
    ckpt_save_path = ckpt_manager.save()
    print('Training finished. Saving checkpoint for {} epoch at {}'.format(epochs, ckpt_save_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file containing training parameters", type=str)
    args = parser.parse_args()
    user_config = load_file(args.config)
    do_training(user_config)


if __name__ == "__main__":
    main()
