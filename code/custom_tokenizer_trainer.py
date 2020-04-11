import os
import json
import tokenizers
import argparse
import sys


class CustomTokenizerTrainer():
    """CustomTokenizerTrainer based on Byte-Level BPE Tokenizer

    Byte-level BPE was introduced by OpenAI with their GPT-2 model
    """

    def __init__(self, save_tokenizer_path: str,
                 training_files,
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
        # train tokenizer
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
            "model_type": model_type,
            "vocab_size": self.VOCAB_SIZE
        }

        config_path = os.path.join(self.save_tokenizer_path, "config.json")
        with open(config_path, 'w') as fp:
            json.dump(config, fp)

        # tokenizer_config = {
        #     "max_len": 512
        # }
        #
        # config_path = os.path.join(self.save_tokenizer_path, "tokenizer_config.json")
        # with open(config_path, 'w') as fp:
        #     json.dump(tokenizer_config, fp)

        special_tokens_map = {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "sep_token": "</s>",
                              "pad_token": "<pad>", "cls_token": "<s>", "mask_token": "<mask>"}

        config_path = os.path.join(self.save_tokenizer_path, "special_tokens_map.json")
        with open(config_path, 'w') as fp:
            json.dump(special_tokens_map, fp)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_size", default=80000, help="Vocabulary Size", type=int)
    parser.add_argument("--lowercase", type=str2bool, default="False")
    parser.add_argument("--min_frequency", default=2, type=int,
                        help="Minimum frequency to consider while training tokenizer")
    parser.add_argument("--training_folder", default=None, help="Folder containing files for tokenizer")
    parser.add_argument("--path_to_save_tokenizer", default=None, help="Folder to save tokenizer files")
    args = parser.parse_args()
    print("Args: ", sys.argv[1:])

    # order of special tokens is important for indexing. Let pad be token_id=0
    special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
    training_files = os.listdir(args.training_folder)
    # remove OS specific files from the training_folder
    try:
        training_files.remove(".DS_Store")
    except ValueError:
        pass

    training_files = [os.path.abspath(os.path.join(args.training_folder, x)) for x in training_files]
    # create custom tokenizer
    train_tokenizer_obj = CustomTokenizerTrainer(args.path_to_save_tokenizer,
                                                 training_files,
                                                 special_tokens,
                                                 args.min_frequency,
                                                 args.lowercase,
                                                 args.vocab_size)
    # train your Byte Pair Tokenizer on your own training files!
    train_tokenizer_obj.train()

    '''
    # how to load the saved tokenizer
    from transformers import AutoTokenizer

    # make sure the path contains 3 files: config.json, merges.txt and vocab.json
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_path, cache_dir=None)

    # sample usage
    text = "Montreal is a great city".strip().lowercase()
    tokenizer.tokenize(text)
    '''


if __name__ == "__main__":
    main()
