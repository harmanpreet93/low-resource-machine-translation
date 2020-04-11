import tensorflow as tf
import io


class DataLoader:

    def __init__(self, batch_size, input_lang, target_lang, tokenizer_en, tokenizer_fr, shuffle=True):
        self.BATCH_SIZE = batch_size
        self.input_lang = input_lang
        self.target_lang = target_lang
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.BUFFER_SIZE = 10000  # size to shuffle
        self.shuffle = shuffle
        self.initialize()

    def initialize(self):
        # read files
        # TODO: check if encoding='UTF-8' or 'latin-1' argument should be passed?
        # TODO: Add pre-processing steps for English language if not done already before this point: removing punctuation

        aligned_sentences_en = io.open(self.input_lang).read().strip().split('\n')
        # tokenize automatically add special tokens, then pad it to max length
        padded_sequences_en = [self.tokenizer_en.encode(sentence.lower())["input_ids"] for sentence in
                               aligned_sentences_en]

        # self.aligned_path_fr can be None while testing
        if self.target_lang is not None:
            aligned_sentences_fr = io.open(self.target_lang).read().strip().split('\n')
            padded_sequences_fr = [self.tokenizer_fr.encode(sentence)["input_ids"] for sentence in aligned_sentences_fr]
        else:
            aligned_sentences_fr = [None] * len(aligned_sentences_en)
            padded_sequences_fr = [None] * len(padded_sequences_en)

        # both input and target should have same number of examples
        assert len(padded_sequences_en) == len(padded_sequences_fr)

        # aligned_sentences_fr is required for evaluation (ignore for training)
        if self.shuffle:
            self.data_loader = tf.data.Dataset.from_tensor_slices(
                (padded_sequences_en, padded_sequences_fr, aligned_sentences_fr)).cache().shuffle(
                self.BUFFER_SIZE).batch(
                self.BATCH_SIZE).prefetch(
                tf.data.experimental.AUTOTUNE)
        else:
            self.data_loader = tf.data.Dataset.from_tensor_slices(
                (padded_sequences_en, padded_sequences_fr, aligned_sentences_fr)).cache().batch(
                self.BATCH_SIZE).prefetch(
                tf.data.experimental.AUTOTUNE)

    def get_data_loader(self):
        '''
        Returns: ``tf.data.Dataset`` object
        '''
        return self.data_loader
