import argparse
import subprocess
import tempfile


def generate_predictions(input_file_path: str, pred_file_path: str):
    """Generates predictions for the machine translation task (EN->FR).

    You are allowed to modify this function as needed, but one again, you cannot
    modify any other part of this file. We will be importing only this function
    in our final evaluation script. Since you will most definitely need to import
    modules for your code, you must import these inside the function itself.

    Args:
        input_file_path: the file path that contains the input data.
        pred_file_path: the file path where to store the predictions.

    Returns: None

    """

    ##### MODIFY BELOW #####
    from generate_model_predictions import do_evaluation

    # config file containing parameters required for model: change model paths, tokenizer paths, seq lengths
    user_config = '/project/cq-training-1/project2/submissions/team08/code/user_config.json'

    # generate translations
    do_evaluation(user_config,
                  input_file_path,
                  target_file_path=None,
                  pred_file_path=pred_file_path)
    ##### MODIFY ABOVE #####


def compute_bleu(pred_file_path: str, target_file_path: str, print_all_scores: bool):
    """

    Args:
        pred_file_path: the file path that contains the predictions.
        target_file_path: the file path that contains the targets (also called references).
        print_all_scores: if True, will print one score per example.

    Returns: None

    """
    out = subprocess.run(["sacrebleu", "--input", pred_file_path, target_file_path, '--tokenize',
                          'none', '--sentence-level', '--score-only'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = out.stdout.split(b'\n')
    if print_all_scores:
        print('\n'.join(lines[:-1]))
        return -1
    else:
        scores = [float(x) for x in lines[:-1]]
        print('final avg bleu score: {:.2f}'.format(sum(scores) / len(scores)))
        return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser('script for evaluating a model.')
    parser.add_argument('--target-file-path', help='path to target (reference) file', required=True)
    parser.add_argument('--input-file-path', help='path to input file', required=True)
    parser.add_argument('--print-all-scores', help='will print one score per sentence',
                        action='store_true')
    parser.add_argument('--do-not-run-model',
                        help='will use --input-file-path as predictions, instead of running the '
                             'model on it',
                        action='store_true')

    args = parser.parse_args()

    if args.do_not_run_model:
        compute_bleu(args.input_file_path, args.target_file_path, args.print_all_scores)
    else:
        _, pred_file_path = tempfile.mkstemp()
        generate_predictions(args.input_file_path, pred_file_path)
        compute_bleu(pred_file_path, args.target_file_path, args.print_all_scores)


if __name__ == '__main__':
    main()
