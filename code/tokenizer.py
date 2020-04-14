import argparse
import logging
import ntpath
import os
import re

import spacy
import tqdm

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser('script for tokenizing data. It relies on the spacy tokenizer')
    parser.add_argument('--input', nargs='+', help='input file. Note it can be more than one',
                        required=True)
    parser.add_argument('--output', help='path to outputs - will store files here',
                        required=True)
    parser.add_argument('--lang', help='either en or fr', required=True)
    parser.add_argument('--keep-case', help='will not lowercase', action='store_true')
    parser.add_argument('--keep-empty-lines', help='will keep empty lines', action='store_true')
    parser.add_argument('--newline-to-space', help='converts newlines to spaces',
                        action='store_true')
    parser.add_argument('--skip-lines-with-pattern', nargs='*', default=[],
                        help='skip lines where any of these regex applies')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    logger.info('will keep the case? {}'.format(args.keep_case))
    logger.info('will keep empty lines? {}'.format(args.keep_empty_lines))
    logger.info('will convert newlines to space? {}'.format(args.newline_to_space))
    logger.info('will skip lines with the following regex: {}'.format(args.skip_lines_with_pattern))

    if args.lang == 'en':
        try:
            tokenizer = spacy.load("en_core_web_sm")
        except OSError:
            raise ValueError('please run the following to download the model: '
                             'python -m spacy download en_core_web_sm')
    elif args.lang == 'fr':
        try:
            tokenizer = spacy.load("fr_core_news_sm")
        except OSError:
            raise ValueError('please run the following to download the model: '
                             'python -m spacy download fr_core_news_sm')
    else:
        raise ValueError('lang {} not supported'.format(args.lang))

    regs = []
    for reg in args.skip_lines_with_pattern:
        regs.append(re.compile(reg))

    done = 0
    for current_file in args.input:
        logger.info('tokenizing file {}'.format(current_file))
        tot_lines, empty_skipped, regex_skipped= tokenize(
            current_file, args.output, tokenizer, args.keep_case, args.keep_empty_lines,
            args.newline_to_space, regs)
        done += 1
        logger.info('done ({} / {}) - skipped {} lines (over {} - i.e., {:3.2f}%) because empty,'
                    ' skipped {} lines (over {} - i.e., {:3.2f}%) because of regex'.format(
            done, len(args.input), empty_skipped, tot_lines, (empty_skipped / tot_lines) * 100,
            regex_skipped, tot_lines, (regex_skipped / tot_lines) * 100))


def get_stream_size(stream):
    result = sum(1 for _ in stream)
    stream.seek(0)
    return result


def tokenize(current_file, output, tokenizer, keep_case, keep_empty_lines, newline_to_space, regs):
    file_name = ntpath.basename(current_file)
    out_tokenized_path = os.path.join(output, file_name)
    empty_skipped = 0
    regex_skipped = 0
    tot_lines = 0
    separator = ' ' if newline_to_space else '\n'
    with open(current_file, 'r') as stream:
        file_size = get_stream_size(stream)
        with open(out_tokenized_path, 'w') as out_tokenized_stream:
            for line in tqdm.tqdm(stream, total=file_size):
                tot_lines += 1
                if not keep_case:
                    line = line.lower()
                if not keep_empty_lines and line.strip() == '':
                    empty_skipped += 1
                    continue
                if any([reg.match(line) for reg in regs]):
                    regex_skipped += 1
                    continue
                tokens = tokenizer(line.strip())
                out_tokenized_stream.write(' '.join([token.text for token in tokens]) + separator)
    return tot_lines, empty_skipped, regex_skipped


if __name__ == '__main__':
    main()
