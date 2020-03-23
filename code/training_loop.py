import os
import tqdm
import json
import typing
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from data_loader import DataLoader
from model_logging import get_logger, get_summary_writers, do_code_profiling
from tensorboard.plugins.hparams import api as hp
import glob

logger = get_logger()


def k_to_true_ghi(max_k_ghi, k, clearsky_ghi):
    true_ghi = tf.math.multiply_no_nan(k * max_k_ghi, clearsky_ghi)
    return true_ghi


def ghi_to_k(max_k_ghi, true_ghi, clearsky_ghi):
    true_ghi = tf.maximum(true_ghi, 0.0)
    k = tf.math.divide_no_nan(true_ghi, clearsky_ghi * max_k_ghi)
    # Clip too large and small k values
    k = tf.minimum(k, 1.0)
    k = tf.maximum(k, 0.0)
    return k


def mask_nighttime_predictions(*args, daytime_flag):
    weight = tf.reduce_sum(tf.cast(daytime_flag, tf.float32))
    outputs = []
    for arg in args:
        outputs += [tf.boolean_mask(tensor=arg, mask=daytime_flag)]
    return outputs + [weight]


def train_step(model, optimizer, loss_fn, max_k_ghi, x_train, y_train, use_image_data_only):
    k_train = ghi_to_k(max_k_ghi, true_ghi=y_train, clearsky_ghi=x_train[1])
    with tf.GradientTape() as tape:
        k_pred, y_pred = model(x_train, training=True, use_image_data_only=use_image_data_only)
        daytime_flag = tf.squeeze(x_train[3])
        k_train = tf.squeeze(k_train)
        k_pred, k_train, y_pred, y_train, weight = \
            mask_nighttime_predictions(k_pred, k_train, y_pred, y_train, daytime_flag=daytime_flag)
        loss = loss_fn(k_train, k_pred)
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    return loss, y_train, y_pred, weight


def test_step(model, loss_fn, max_k_ghi, x_test, y_test, use_image_data_only):
    k_test = ghi_to_k(max_k_ghi, true_ghi=y_test, clearsky_ghi=x_test[1])
    k_pred, y_pred = model(x_test, use_image_data_only=use_image_data_only)
    daytime_flag = tf.squeeze(x_test[3])
    k_test = tf.squeeze(k_test)
    y_pred, y_test, k_pred, k_test, weight = \
        mask_nighttime_predictions(y_pred, y_test, k_pred, k_test, daytime_flag=daytime_flag)
    loss = loss_fn(k_test, k_pred)
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
        tr_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        val_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        tr_datetimes: typing.List[datetime.datetime],
        val_datetimes: typing.List[datetime.datetime],
        tr_time_offsets: typing.List[datetime.timedelta],
        val_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
        user_config: typing.Dict[typing.AnyStr, typing.Any]
):
    """Trains and saves the model to file"""

    # Import the training and validation data loaders, import the model
    Train_DL = DataLoader(
        dataframe,
        tr_datetimes,
        tr_stations,
        tr_time_offsets,
        user_config,
        data_folder=os.path.expandvars(user_config["train_data_folder"])
    )
    Val_DL = DataLoader(
        dataframe,
        val_datetimes,
        val_stations,
        val_time_offsets,
        user_config,
        data_folder=os.path.expandvars(user_config["val_data_folder"])
    )

    train_data_loader = Train_DL.get_data_loader()
    val_data_loader = Val_DL.get_data_loader()

    # Set random seed before initializing the model weights
    tf.random.set_seed(user_config["random_seed"])

    model = MainModel(tr_stations, tr_time_offsets, user_config)

    # set hyper-parameters
    nb_epoch = user_config["nb_epoch"]
    learning_rate = user_config["learning_rate"]
    max_k_ghi = user_config["max_k_ghi"]

    # Optimizer: Adam - for decaying learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Objective/Loss function: MSE Loss
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Define tensorboard metrics
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    train_rmse = tf.keras.metrics.RootMeanSquaredError()
    test_rmse = tf.keras.metrics.RootMeanSquaredError()

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
            'max_k_ghi': max_k_ghi,
            'batch_size': user_config["batch_size"],
            'input_seq_length': user_config["input_seq_length"],
            'nb_feature_maps': user_config["nb_feature_maps"],
            'nb_dense_units': user_config["nb_dense_units"],
            'use_all_data_at_epoch': user_config["use_all_data_at_epoch"]
        })

    n_train_steps = len(glob.glob(user_config["train_data_folder"] + "/*hdf5"))
    n_val_steps = len(glob.glob(user_config["val_data_folder"] + "/*hdf5"))

    # training starts here
    with tqdm.tqdm("training", total=nb_epoch) as pbar:
        pbar.update(start_epoch)
        for epoch in range(start_epoch, nb_epoch):

            current_train_steps_start_point = epoch * n_train_steps
            current_val_steps_start_point = epoch * n_val_steps

            with tqdm.tqdm("Train steps", total=n_train_steps) as train_pbar:

                # Train the model using the training set for one epoch
                for i, minibatch in enumerate(train_data_loader):
                    loss, y_train, y_pred, weight = train_step(
                        model,
                        optimizer,
                        loss_fn,
                        max_k_ghi,
                        x_train=minibatch[:-1],
                        y_train=minibatch[-1],
                        use_image_data_only=(epoch < user_config["use_all_data_at_epoch"])
                    )
                    train_loss(loss, sample_weight=weight)
                    train_rmse(y_train, y_pred, sample_weight=weight)

                    train_pbar.update(1)

                    if i % 10 == 0:
                        with train_step_writer.as_default():
                            tf.summary.scalar('train step loss', train_loss.result(),
                                              step=current_train_steps_start_point + i)
                            tf.summary.scalar('train step rmse', train_rmse.result(),
                                              step=current_train_steps_start_point + i)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('rmse', train_rmse.result(), step=epoch)

            with tqdm.tqdm("Validation steps", total=n_val_steps) as val_pbar:

                # Evaluate model performance on the validation set after training for one epoch
                for j, minibatch in enumerate(val_data_loader):
                    loss, y_test, y_pred, weight = test_step(
                        model,
                        loss_fn,
                        max_k_ghi,
                        x_test=minibatch[:-1],
                        y_test=minibatch[-1],
                        use_image_data_only=(epoch < user_config["use_all_data_at_epoch"])
                    )
                    test_loss(loss, sample_weight=weight)
                    test_rmse(y_test, y_pred, sample_weight=weight)

                    val_pbar.update(1)

                    if j % 10 == 0:
                        with test_step_writer.as_default():
                            tf.summary.scalar('val step loss', test_loss.result(),
                                              step=current_val_steps_start_point + j)
                            tf.summary.scalar('val step rmse', test_rmse.result(),
                                              step=current_val_steps_start_point + j)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('rmse', test_rmse.result(), step=epoch)

            with hparam_summary_writer.as_default():
                tf.summary.scalar("val_loss", test_loss.result(), step=epoch)
                tf.summary.scalar("val_rmse", test_rmse.result(), step=epoch)

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

            logger.debug(
                "Epoch {0}/{1}, Train RMSE = {2}, Val RMSE = {3}"
                .format(epoch + 1, nb_epoch, train_rmse.result(), test_rmse.result())
            )

            # Reset metrics every epoch
            train_loss.reset_states()
            train_rmse.reset_states()
            test_loss.reset_states()
            test_rmse.reset_states()

            pbar.update(1)
