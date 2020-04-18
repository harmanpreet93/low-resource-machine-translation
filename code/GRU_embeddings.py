import numpy as np
import argparse
import utils_GRU
from gensim.models import Word2Vec
from gensim.models import FastText


# tokenization and punctuation removal
# already done with the script provided by the TA for both unaligned corpus
# these files are in the data_folder of the GRU_config.json file

def list_of_examples(GRU_config, lang):
    """
    creates a list for each example in the input file
    """
    words = []
    if lang == "en":
        with open(GRU_config["data_folder"] + "tokenized_nopunc_unaligned.en", "r", encoding="UTF-8") as f:
            for line in f:
                line = line.strip()
                words.append(line.split(" "))
    elif lang == "fr":
        with open(GRU_config["data_folder"] + "tokenized_nopunc_unaligned.fr", "r", encoding="UTF-8") as f:
            for line in f:
                line = line.strip()
                words.append(line.split(" "))
    else:
        raise Exception("Unsupported lang argument")
    return words


def make_embeddings(GRU_config, embedding_model, lang):
    """
    Saves word vectors trained by an embedding model, either Word2Vec or FastText
    """
    if lang == "en":
        sentences = list_of_examples(GRU_config, "en")
        if embedding_model == "Word2Vec":
            w2v = Word2Vec(
                size=GRU_config["word_embedding_dim"], window=2, min_count=1, seed=GRU_config["random_seed"])
            w2v.build_vocab(sentences)
            w2v.train(sentences, total_examples=len(sentences), epochs=5)
            word_vectors = w2v.wv
            word_vectors.save(GRU_config["word_embeddings_path"] + "w2v.en")
        elif embedding_model == "FastText":
            fasttext = FastText(size=GRU_config["word_embedding_dim"], window=2,
                                min_count=1, seed=6759)
            fasttext.build_vocab(sentences)
            fasttext.train(sentences, total_examples=len(sentences), epochs=5)
            word_vectors = fasttext.wv
            word_vectors.save(GRU_config["word_embeddings_path"] + "fast.en")
        else:
            raise Exception("Unsupported model name")
    elif lang == "fr":
        sentences = list_of_examples(GRU_config, "fr")
        if embedding_model == "Word2Vec":
            w2v = Word2Vec(
                size=GRU_config["word_embedding_dim"], window=2, min_count=1, seed=GRU_config["random_seed"])
            w2v.build_vocab(sentences)
            w2v.train(sentences, total_examples=len(sentences), epochs=5)
            word_vectors = w2v.wv
            word_vectors.save(GRU_config["word_embeddings_path"] + "w2v.fr")
        elif embedding_model == "FastText":
            fasttext = FastText(size=GRU_config["word_embedding_dim"], window=2,
                                min_count=1, seed=6759)
            fasttext.build_vocab(sentences)
            fasttext.train(sentences, total_examples=len(sentences), epochs=5)
            word_vectors = fasttext.wv
            word_vectors.save(GRU_config["word_embeddings_path"] + "fast.fr")
        else:
            raise Exception("Unsupported embedding model name")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Configuration file containing training parameters",
        type=str)
    parser.add_argument(
        "--embedding_model",
        help="Embedding model to be ran, either Word2Vec or FastText",
        type=str)
    parser.add_argument(
        "--lang",
        help="Language for the embedding, either 'en' or 'fr'",
        type=str)
    args = parser.parse_args()
    GRU_config = utils_GRU.load_file(args.config)
    make_embeddings(GRU_config, args.embedding_model, args.lang)


if __name__ == "__main__":
    main()
