import time
import argparse
from utils import *
from data_loader import DataLoader
from generate_model_predictions import sacrebleu_metric, compute_bleu


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


def val_step(model, loss_object, inp, tar,
             val_loss, val_accuracy, pad_token_id):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    predictions, _ = model(inp, tar_inp,
                           False,
                           enc_padding_mask,
                           combined_mask,
                           dec_padding_mask)
    loss = loss_function(tar_real, predictions, loss_object, pad_token_id)

    val_loss(loss)
    val_accuracy(tar_real, predictions)


def compute_bleu_score(transformer_model, dataset, user_config, tokenizer_tar, epoch):
    inp_language = user_config["inp_language"]
    target_language = user_config["target_language"]
    checkpoint_path = user_config["transformer_checkpoint_path"]
    val_aligned_path_tar = user_config["val_data_path_{}".format(target_language)]
    pred_file_path = "../log/log_{}_{}/".format(inp_language, target_language) + checkpoint_path.split('/')[
        -1] + "_epoch-" + str(epoch) + "_prediction_{}.txt".format(target_language)

    sacrebleu_metric(transformer_model, pred_file_path, None,
                     tokenizer_tar, dataset,
                     tokenizer_tar.MAX_LENGTH)
    print("-----------------------------")
    scores = compute_bleu(pred_file_path, val_aligned_path_tar, print_all_scores=False)
    print("-----------------------------")

    # append checkpoint and score to file name for easy reference
    new_path = "../log/log_{}_{}/".format(inp_language, target_language) + checkpoint_path.split('/')[
        -1] + "_epoch-" + str(epoch) + "_prediction_{}_{:.2f}".format(target_language, scores) + ".txt"
    # append score and checkpoint name to file_name
    os.rename(pred_file_path, new_path)
    print("Saved translated prediction at {}".format(new_path))


def do_training(user_config):
    inp_language = user_config["inp_language"]
    target_language = user_config["target_language"]

    print("\n****Training model from {} to {}****\n".format(inp_language, target_language))

    print("****Loading tokenizers****")
    # load pre-trained tokenizer
    tokenizer_inp, tokenizer_tar = load_tokenizers(inp_language, target_language, user_config)

    print("****Loading train dataset****")
    # train data loader
    train_aligned_path_inp = user_config["train_data_path_{}".format(inp_language)]
    train_aligned_path_tar = user_config["train_data_path_{}".format(target_language)]
    train_dataloader = DataLoader(user_config["transformer_batch_size"],
                                  train_aligned_path_inp,
                                  train_aligned_path_tar,
                                  tokenizer_inp,
                                  tokenizer_tar,
                                  inp_language,
                                  target_language,
                                  True)
    train_dataset = train_dataloader.get_data_loader()

    print("****Loading val dataset****")
    # val data loader
    val_aligned_path_inp = user_config["val_data_path_{}".format(inp_language)]
    val_aligned_path_tar = user_config["val_data_path_{}".format(target_language)]
    val_dataloader = DataLoader(user_config["transformer_batch_size"] * 2,  # for fast validation increase batch size
                                val_aligned_path_inp,
                                val_aligned_path_tar,
                                tokenizer_inp,
                                tokenizer_tar,
                                inp_language,
                                target_language,
                                False)
    val_dataset = val_dataloader.get_data_loader()

    # define loss and accuracy metrics
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    print("****Loading transformer model****")
    # load model and optimizer
    transformer_model, optimizer, ckpt_manager = load_transformer_model(user_config, tokenizer_inp, tokenizer_tar)

    epochs = user_config["transformer_epochs"]
    print("\nTraining model now...")
    for epoch in range(epochs):
        print()
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        # inp -> english, tar -> french
        for (batch, (inp, tar, _)) in enumerate(train_dataset):
            train_step(transformer_model, loss_object, optimizer, inp, tar,
                       train_loss, train_accuracy, pad_token_id=tokenizer_tar.pad_token_id)

            if batch % 50 == 0:
                print('Train: Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

            if (batch + 1) % 2200 == 0:
                # inp -> english, tar -> french
                for (batch_, (inp, tar, _)) in enumerate(val_dataset):
                    val_step(transformer_model, loss_object, inp, tar,
                             val_loss, val_accuracy, pad_token_id=tokenizer_tar.pad_token_id)
                print('Val Loss: {:.4f}, Val Accuracy: {:.4f}\n'.format(val_loss.result(), val_accuracy.result()))
                if user_config["compute_bleu"]:
                    print("\nComputing BLEU at batch {}: ".format(batch))
                    compute_bleu_score(transformer_model, val_dataset, user_config, tokenizer_tar, epoch + 1)

        print("After {} epochs".format(epoch + 1))
        print('Train Loss: {:.4f}, Train Accuracy: {:.4f}'.format(train_loss.result(), train_accuracy.result()))

        # inp -> english, tar -> french
        for (batch, (inp, tar, _)) in enumerate(val_dataset):
            val_step(transformer_model, loss_object, inp, tar,
                     val_loss, val_accuracy, pad_token_id=tokenizer_tar.pad_token_id)
        print('Val Loss: {:.4f}, Val Accuracy: {:.4f}'.format(val_loss.result(), val_accuracy.result()))

        print('Time taken for training epoch {}: {} secs'.format(epoch + 1, time.time() - start))

        # evaluate and save model every x-epochs
        # if epoch % 2 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint after epoch {} at {}'.format(epoch + 1, ckpt_save_path))
        if user_config["compute_bleu"]:
            print("\nComputing BLEU at epoch {}: ".format(epoch + 1))
            compute_bleu_score(transformer_model, val_dataset, user_config, tokenizer_tar, epoch + 1)

    # # save model after last epoch
    # ckpt_save_path = ckpt_manager.save()
    # print('*****Training finished. Saving checkpoint for {} epoch at {}*****'.format(epochs, ckpt_save_path))
    #
    # # compute bleu score when training finished, and if bleu score wasn't already computed
    # if user_config["compute_bleu"] and (epochs - 1) % 2 != 0:
    #     print("\nComputing BLEU after training finished at epoch: {}: ".format(epochs))
    #     compute_bleu_score(transformer_model, val_dataset, user_config, tokenizer_tar, epochs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file containing training parameters", type=str)
    args = parser.parse_args()
    user_config = load_file(args.config)
    seed = user_config["random_seed"]
    set_seed(seed)
    print(json.dumps(user_config, indent=2))
    do_training(user_config)


if __name__ == "__main__":
    main()
