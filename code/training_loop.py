import os
import tqdm
import json
import typing
import datetime
import numpy as np
import tensorflow as tf
from data_loader import DataLoader
from model_logging import get_logger, get_summary_writers, do_code_profiling
from tensorboard.plugins.hparams import api as hp
import glob
from transformers import BertConfig, TFBertForMaskedLM

logger = get_logger()

def mask_vals(*args, mask_flag): #TODO
    weight = tf.reduce_sum(tf.cast(mask_flag, tf.float32))
    outputs = []
    for arg in args:
        outputs += [tf.boolean_mask(tensor=arg, mask=mask_flag)]
    return outputs + [weight]


def train_step(model, optimizer, loss_fn, x_train, y_train, mask_targets):
    with tf.GradientTape() as tape:
        k_pred, y_pred = model(x_train, training=True)
        y_pred, y_train, weight = mask_vals(y_pred, y_train, mask_flag=mask_targets)
        loss = loss_fn(y_pred, y_train)
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    return loss, y_train, y_pred, weight


def test_step(model, loss_fn, x_test, y_test, mask_targets):
    k_pred, y_pred = model(x_test)
    y_pred, y_test, weight = mask_vals(y_pred, y_test, mask_flag=mask_targets)
    loss = loss_fn(y_pred, y_test)
    return loss, y_test, y_pred, weight


def manage_model_start_time(ignore_checkpoints):
    model_metadata_path = '../model/model_metadata.json'
    if os.path.isfile(model_metadata_path) and not ignore_checkpoints:
        # Metadata found; log training with previous timestamp
        with open(model_metadata_path, "r") as fd:
            model_train_start_time = json.load(fd)["model_train_start_time"]
            return model_train_start_time

    # No file found; log training with current timestamp
    model_train_start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(model_metadata_path, 'w') as outfile:
        json.dump({"model_train_start_time": model_train_start_time}, outfile, indent=2)
    return model_train_start_time


def manage_model_checkpoints(optimizer, model, user_config):
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, '../model/tf_ckpts', max_to_keep=3)

    if user_config["ignore_checkpoints"]:
        print("Model checkpoints ignored; Initializing from scratch.")
        early_stop_metric = np.inf
        np.save(user_config["model_info"], [early_stop_metric])
        model_train_start_time = manage_model_start_time(ignore_checkpoints=True)
    else:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored model from {}".format(manager.latest_checkpoint))
            model_train_start_time = manage_model_start_time(ignore_checkpoints=False)
            early_stop_metric = np.load(user_config["model_info"])[0]
        else:
            print("No checkpoint found; Initializing from scratch.")
            model_train_start_time = manage_model_start_time(ignore_checkpoints=True)
            early_stop_metric = np.inf

    start_epoch = ckpt.step.numpy()

    return manager, ckpt, early_stop_metric, start_epoch, model_train_start_time


@do_code_profiling
def train(
        MainModel,
        user_config: typing.Dict[typing.AnyStr, typing.Any]
):
    """Trains and saves the model to file"""

    # Import the training and validation data loaders, import the model
    Train_DL = DataLoader(
        user_config,
        data_folder=os.path.expandvars(user_config["train_data_folder"])
    )
    Val_DL = DataLoader(
        user_config,
        data_folder=os.path.expandvars(user_config["val_data_folder"])
    )

    train_data_loader = Train_DL.get_data_loader()
    val_data_loader = Val_DL.get_data_loader()

    # Set random seed before initializing the model weights
    tf.random.set_seed(user_config["random_seed"])

    config = BertConfig.from_pretrained('../code/bert_config_tiny.json')
    model = TFBertForMaskedLM(config)

    # set hyper-parameters
    nb_epoch = user_config["nb_epoch"]
    learning_rate = user_config["learning_rate"]

    # Optimizer: Adam - for decaying learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)

    # Objective/Loss function: MSE Loss
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Define tensorboard metrics
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    # Checkpoint management (for model save/restore)
    manager, ckpt, early_stop_metric, start_epoch, start_time = manage_model_checkpoints(optimizer, model, user_config)

    # Get tensorboard file writers
    train_summary_writer, test_summary_writer, hparam_summary_writer, train_step_writer, test_step_writer = \
        get_summary_writers(start_time)

    # Log hyperparameters
    with hparam_summary_writer.as_default():
        hp.hparams({
            'nb_epoch': nb_epoch,
            'learning_rate': learning_rate,
            'batch_size': user_config["batch_size"],
            'input_seq_length': user_config["input_seq_length"],
        })

    # training starts here
    with tqdm.tqdm("training", total=nb_epoch) as pbar:
        pbar.update(start_epoch)
        for epoch in range(start_epoch, nb_epoch):

            with tqdm.tqdm("Train steps", total=n_train_steps) as train_pbar:

                # Train the model using the training set for one epoch
                for i, minibatch in enumerate(train_data_loader):
                    loss, y_train, y_pred, weight = train_step(
                        model,
                        optimizer,
                        loss_fn,
                        x_train=minibatch[:-1],
                        y_train=minibatch[-1]
                    )
                    train_loss(loss, sample_weight=weight)

                    train_pbar.update(1)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)

            with tqdm.tqdm("Validation steps", total=n_val_steps) as val_pbar:

                # Evaluate model performance on the validation set after training for one epoch
                for j, minibatch in enumerate(val_data_loader):
                    loss, y_test, y_pred, weight = test_step(
                        model,
                        loss_fn,
                        x_test=minibatch[:-1],
                        y_test=minibatch[-1]
                    )
                    test_loss(loss, sample_weight=weight)

                    val_pbar.update(1)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)

            with hparam_summary_writer.as_default():
                tf.summary.scalar("val_loss", test_loss.result(), step=epoch)

            # Create a model checkpoint after each epoch
            ckpt.step.assign_add(1)
            save_path = manager.save()
            logger.debug("Saved checkpoint for epoch {}: {}".format(int(ckpt.step), save_path))

            # Save the best model
            if test_loss.result() < early_stop_metric:
                early_stop_metric = test_loss.result()
                model.save_weights("../model/my_model", save_format="tf")
                np.save(user_config["model_info"], [early_stop_metric.numpy()])

            logger.debug(
                "Epoch {0}/{1}, Train Loss = {2}, Val Loss = {3}"
                .format(epoch + 1, nb_epoch, train_loss.result(), test_loss.result())
            )

            # Reset metrics every epoch
            train_loss.reset_states()
            test_loss.reset_states()

            pbar.update(1)
