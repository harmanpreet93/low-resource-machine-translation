import argparse
import logging
import ntpath
import os

import tqdm

logger = logging.getLogger(__name__)

PUNCTUATION = {",", ";", ":", "!", "?", ".", "'", '"', "(", ")", "...", "[", "]", "{", "}"}


def main():
    parser = argparse.ArgumentParser(
        'script to remove punctuation. Data must be already tokenized.')
    parser.add_argument('--input', nargs='+', help='input file. Note it can be more than one')
    parser.add_argument('--output', help='path to outputs - will store files here',
                        required=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for current_file in args.input:
        logger.info('tokenizing file {}'.format(current_file))
        tot_lines, removed_punctuations = remove_punctuation(current_file, args.output)
        logger.info('done - parsed {} lines and removed {} (punctuation) symbols'.format(
            tot_lines, removed_punctuations))


def remove_punctuation(current_file, output):
    file_name = ntpath.basename(current_file)
    out_tokenized_path = os.path.join(output, file_name)
    tot_lines = 0
    removed_punctuations = 0
    with open(current_file, 'r') as stream:
        with open(out_tokenized_path, 'w') as out_tokenized_stream:
            for line in tqdm.tqdm(stream):
                tot_lines += 1
                tokens = line.strip().split()
                filtered_tokens = [token for token in tokens if token not in PUNCTUATION]
                out_tokenized_stream.write(' '.join(filtered_tokens) + '\n')
                removed_punctuations += (len(tokens) - len(filtered_tokens))
    return tot_lines, removed_punctuations


if __name__ == '__main__':
    main()
