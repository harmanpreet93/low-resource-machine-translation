import logging
from datetime import datetime
import tensorflow as tf

logger = None


def get_logger():
    global logger
    if not logger:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        file_name = '../log/' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.log'
        line_format = "%(asctime)s %(levelname)s: %(filename)s:%(funcName)s():%(lineno)d - %(message)s"
        logging.basicConfig(
            level=logging.DEBUG,
            format=line_format,
            filename=file_name,
            filemode="w"
        )
        logger = logging.getLogger(__name__)
        logger.info("Logger initialized")
    return logger


def get_summary_writers(current_time):
    train_log_dir = '../log/gradient_tape/' + current_time + '/train'
    test_log_dir = '../log/gradient_tape/' + current_time + '/test'
    hparam_log_dir = '../log/hparam_tuning/' + current_time + '/hparam'

    train_log_dir_steps = '../log/gradient_tape/' + current_time + '/train_steps'
    test_log_dir_steps = '../log/gradient_tape/' + current_time + '/test_steps'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    hparam_summary_writer = tf.summary.create_file_writer(hparam_log_dir)
    train_summary_writer_steps = tf.summary.create_file_writer(train_log_dir_steps)
    test_summary_writer_steps = tf.summary.create_file_writer(test_log_dir_steps)

    return train_summary_writer, test_summary_writer, hparam_summary_writer, \
        train_summary_writer_steps, test_summary_writer_steps


def do_code_profiling(function):
    def wrapper(*args, **kwargs):
        if args[-1]["code_profiling_enabled"]:
            import cProfile
            import pstats
            profile = cProfile.Profile()
            profile.enable()

            x = function(*args, **kwargs)

            profile.disable()
            profile.dump_stats("../log/profiling_results.prof")
            with open("../log/profiling_results.txt", "w") as f:
                ps = pstats.Stats("../log/profiling_results.prof", stream=f)
                ps.sort_stats('cumulative')
                ps.print_stats()
            return x
        else:
            return function(*args, **kwargs)

    return wrapper
