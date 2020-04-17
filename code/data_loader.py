import tensorflow as tf
import io


class DataLoader:

    def __init__(self, batch_size, input_lang_path, target_lang_path, tokenizer_inp, tokenizer_tar, input_lang,
                 target_lang, shuffle=True):
        self.BATCH_SIZE = batch_size
        self.input_lang_path = input_lang_path
        self.target_lang_path = target_lang_path
        self.tokenizer_inp = tokenizer_inp
        self.tokenizer_tar = tokenizer_tar
        self.BUFFER_SIZE = 60000  # buffer size to shuffle
        self.shuffle = shuffle
        self.input_lang = input_lang
        self.target_lang = target_lang
        self.initialize()

    def initialize(self):
        # read files
        # TODO: Add pre-processing steps for English language if not done already before this point

        aligned_sentences_inp = io.open(self.input_lang_path).read().strip().split('\n')
        # tokenizer automatically add special tokens, then pad it to max length
        if self.input_lang == "en":
            padded_sequences_inp = [self.tokenizer_inp.encode(sentence.lower())["input_ids"] for sentence in
                                    aligned_sentences_inp]
        # lowercase only english characters
        elif self.input_lang == "fr":
            padded_sequences_inp = [self.tokenizer_inp.encode(sentence)["input_ids"] for sentence in
                                    aligned_sentences_inp]
        else:
            raise Exception("Unsupported input language in dataloader: {}".format(self.input_lang))

        # self.aligned_path_fr can be None while testing
        if self.target_lang_path is not None:
            aligned_sentences_tar = io.open(self.target_lang_path).read().strip().split('\n')
            if self.target_lang == "fr":
                padded_sequences_tar = [self.tokenizer_tar.encode(sentence)["input_ids"] for sentence in
                                        aligned_sentences_tar]
            elif self.target_lang == "en":
                padded_sequences_tar = [self.tokenizer_tar.encode(sentence.lower())["input_ids"] for sentence in
                                        aligned_sentences_tar]
            else:
                raise Exception("Unsupported target language in dataloader: {}".format(self.target_lang))

        else:
            aligned_sentences_tar = [self.tokenizer_tar.pad_token_id] * len(aligned_sentences_inp)
            padded_sequences_tar = [self.tokenizer_tar.pad_token_id] * len(padded_sequences_inp)

        # both input and target should have same number of examples
        assert len(padded_sequences_inp) == len(padded_sequences_tar)

        # aligned_sentences_tar is required for evaluation (ignore for training)
        if self.shuffle:
            self.data_loader = tf.data.Dataset.from_tensor_slices(
                (padded_sequences_inp, padded_sequences_tar, aligned_sentences_tar)).shuffle(
                self.BUFFER_SIZE).batch(
                self.BATCH_SIZE).prefetch(
                tf.data.experimental.AUTOTUNE)
        else:
            self.data_loader = tf.data.Dataset.from_tensor_slices(
                (padded_sequences_inp, padded_sequences_tar, aligned_sentences_tar)).batch(
                self.BATCH_SIZE).prefetch(
                tf.data.experimental.AUTOTUNE)

    def get_data_loader(self):
        '''
        Returns: ``tf.data.Dataset`` object
        '''
        return self.data_loader
