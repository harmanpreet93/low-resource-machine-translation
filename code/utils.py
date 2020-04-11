import os
import json
from pretrained_tokenizer import Tokenizer


def load_file(path):
    assert os.path.isfile(path), f"invalid config file: {path}"
    with open(path, "r") as fd:
        return json.load(fd)


def load_tokenizers(inp_language, target_language, user_config):
    # load pre-trained tokenizer
    pretrained_tokenizer_path_inp = user_config["tokenizer_path_{}".format(inp_language)]
    pretrained_tokenizer_path_tar = user_config["tokenizer_path_{}".format(target_language)]

    tokenizer_inp = Tokenizer(inp_language, pretrained_tokenizer_path_inp,
                              max_length=user_config["max_length_{}".format(inp_language)])
    tokenizer_tar = Tokenizer(target_language, pretrained_tokenizer_path_tar,
                              max_length=user_config["max_length_{}".format(target_language)])

    return tokenizer_inp, tokenizer_tar