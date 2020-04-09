import tensorflow as tf
from model_logging import get_logger
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io


class DataLoader():

    def __init__(
            self,
            batch_size,
            aligned_path_en,
            aligned_path_fr,
            tokenizer_en,
            tokenizer_fr
    ):
        self.BATCH_SIZE = batch_size
        self.aligned_path_en = aligned_path_en
        self.aligned_path_fr = aligned_path_fr
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.MAX_LENGTH = self.tokenizer_en.max_len  # 512
        self.initialize()

    def initialize(self):
        self.logger = get_logger()
        self.logger.debug("Initialize start")

        # read files
        # TODO: check if encoding='UTF-8' argument should be passed or not
        aligned_sentences_en = io.open(self.aligned_path_en).read().strip().split('\n')
        aligned_sentences_fr = io.open(self.aligned_path_fr).read().strip().split('\n')

        # tokenize and automatically add
        encoded_sequences_en = [self.tokenizer_en.encode(sentence.lower()) for sentence in aligned_sentences_en]
        encoded_sequences_fr = [self.tokenizer_fr.encode(sentence) for sentence in aligned_sentences_fr]

        padded_sequences_en = pad_sequences(encoded_sequences_en,
                                            padding='post',
                                            value=self.tokenizer_en.pad_token_id,
                                            maxlen=self.MAX_LENGTH)

        padded_sequences_fr = pad_sequences(encoded_sequences_fr,
                                            padding='post',
                                            value=self.tokenizer_fr.pad_token_id,
                                            maxlen=self.MAX_LENGTH)

        self.data_loader = tf.data.Dataset.from_tensor_slices((padded_sequences_en, padded_sequences_fr)).shuffle(
            self.BATCH_SIZE).batch(self.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    def get_data_loader(self):
        '''
        Returns:
            A ``tf.data.Dataset`` object that can be used to produce input tensors for your model. One tensor
            must correspond to one sequence of past imagery data. The tensors must be generated in the order given
            by ``target_sequences``.
        '''
        return self.data_loader
