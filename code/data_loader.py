import tensorflow as tf
from model_logging import get_logger
from tensorflow.keras.preprocessing.sequence import pad_sequences


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
        self.MAX_LENGTH = self.tokenizer_en.max_len # 512
        self.initialize()

    def initialize(self):
        self.logger = get_logger()
        self.logger.debug("Initialize start")

        self.data_loader = tf.data.Dataset.from_generator(
            self.data_generator_fn,
            output_types=(tf.int32, tf.int32)
        ).batch(self.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    def encode_sentence(self, sentence, tokenizer, lowercase=True):
        if lowercase:
            encoded_sequence = tokenizer.encode(sentence.lower())
        else:
            encoded_sequence = tokenizer.encode(sentence) # for french

        padded_seq = pad_sequences([encoded_sequence],
                                   padding='post',
                                   value=tokenizer.pad_token_id,
                                   maxlen=self.MAX_LENGTH)

        return padded_seq[0]

    def data_generator_fn(self):
        with open(self.aligned_path_en, 'r') as f_en, open(self.aligned_path_fr, 'r') as f_fr:
            for line_en, line_fr in zip(f_en, f_fr):
                # return inp, out
                yield self.encode_sentence(line_en, self.tokenizer_en, lowercase=True), \
                      self.encode_sentence(line_fr, self.tokenizer_fr, lowercase=False)

    def get_data_loader(self):
        '''
        Returns:
            A ``tf.data.Dataset`` object that can be used to produce input tensors for your model. One tensor
            must correspond to one sequence of past imagery data. The tensors must be generated in the order given
            by ``target_sequences``.
        '''
        return self.data_loader
