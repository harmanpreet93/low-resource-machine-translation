import os
import json
import tokenizers


class CustomTokenizerTrainer():
    """CustomTokenizerTrainer based on ByteLevelBPETokenizer"""

    def __init__(self, save_tokenizer_path: str,
                 training_files: str,
                 special_tokens,
                 min_frequency: int,
                 lowercase: bool,
                 VOCAB_SIZE: int):
        super(CustomTokenizerTrainer, self).__init__()
        self.save_tokenizer_path = save_tokenizer_path
        self.training_files = training_files
        self.special_tokens = special_tokens
        self.min_frequency = min_frequency
        self.lowercase = lowercase
        self.VOCAB_SIZE = VOCAB_SIZE
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(lowercase=self.lowercase)

    def train(self):
        self.tokenizer.train(
            files=self.training_files,
            vocab_size=self.VOCAB_SIZE,
            min_frequency=self.min_frequency,
            show_progress=True,
            special_tokens=self.special_tokens
        )

        if not os.path.exists(self.save_tokenizer_path):
            os.makedirs(self.save_tokenizer_path)

        # This saves 2 files, which are required later by the tokenizer: merges.txt and vocab.json
        self.tokenizer.save(self.save_tokenizer_path)

        model_type = "roberta"  # roBERTa model is better than BERT for language modelling
        config = {
            "vocab_size": self.VOCAB_SIZE,
            "model_type": model_type
        }

        config_path = os.path.join(self.save_tokenizer_path, "config.json")
        with open(config_path, 'w') as fp:
            json.dump(config, fp)


def main():
    # how to train a tokenizer
    data_path = "../../../machine-translation/data"
    VOCAB_SIZE = 60000
    min_frequency = 2
    lowercase = True
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    unaligned_en_path = os.path.join(data_path, 'unaligned.en')
    aligned_en_path = os.path.join(data_path, 'train.lang1')
    training_files = [unaligned_en_path, aligned_en_path]
    pretrained_tokenizer_path = os.path.join(data_path, 'tokenizer_data')

    # create custom tokenizer
    tokenizerObj = CustomTokenizerTrainer(pretrained_tokenizer_path,
                                          training_files,
                                          special_tokens,
                                          min_frequency,
                                          lowercase,
                                          VOCAB_SIZE)

    ALREADY_TRAINED_ONCE = False
    if ALREADY_TRAINED_ONCE:
        print("Tokenizer already trained. Set ALREADY_TRAINED_ONCE=False to re-train tokenizer")
    else:
        # train your Byte Pair Tokenizer on your own training files!
        tokenizerObj.train()

    ##############################################################
    # how to load the saved tokenizer
    from transformers import AutoTokenizer

    # make sure the path contains 3 files: config.json, merges.txt and vocab.json
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_path, cache_dir=None)

    # sample usage
    text = "Montreal is a great city".strip().lowercase()
    tokenizer.tokenize(text)


if __name__ == "__main__":
    main()
