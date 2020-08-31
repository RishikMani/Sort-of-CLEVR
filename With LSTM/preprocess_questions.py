import sys
import os
import argparse
import json
import h5py
import numpy as np

from preprocess import tokenize, encode, build_vocab

sys.path.insert(0, os.path.abspath('.'))

# Pre-processing script for CLEVR question files.
# Creates *_questions.h5 dataset for different datasets.
# Converts all the tokens from all questions and answers to idx.

parser = argparse.ArgumentParser()
parser.add_argument("--preprocess_questions_for", default="train", type=str,
                    help="Please provide in the type of dataset for which you"
                         "want to process questions. The allowed datasets are"
                         "train and test.")
parser.add_argument("--input_questions", default=None, type=str,
                    help="Path to the input questions.")
parser.add_argument("--input_vocab_json", default="",
                    help="Path where the vocabulary from the questions and "
                         "answers would be searched.")
parser.add_argument("--expand_vocab", default=1, type=int,
                    help="Set this parameter to 1 if you want to expand the"
                         "vocabulary in addition to the train dataset"
                         "vocabulary.")
parser.add_argument("--unk_threshold", default=1, type=int)
parser.add_argument("--encode_unk", default=0, type=int)
parser.add_argument("--output_h5_file", default=None, type=str,
                    help="Path where the features derived from questions and "
                         "answers would be written")
parser.add_argument("--output_vocab_json", default="./output/vocab.json",
                    help="Path where the vocabulary from the questions and "
                         "answers would be written.")

output_dir = "./output"

try:
    os.makedirs(output_dir)
except:
    print("directory {} already exists".format(output_dir))


def main(args):
    if (args.input_vocab_json == '') and (args.output_vocab_json == ''):
        print("Must give one of --input_vocab_json or --output_vocab_json")
        return

    print("Loading questions...")
    with open(args.input_questions, 'r') as f:
        questions = f.read()
    questions = questions.split("\n")
    questions = questions[:-1]

    print("Loading answers...")
    with open(args.input_answers, 'r') as f:
        answers = f.read()
    answers = answers.split("\n")
    answers = answers[:-1]

    answer_token_to_idx = None
    # Either create the vocab or load it from disk
    if args.input_vocab_json == "" or args.expand_vocab == 1:
        print("Building vocab...")

        # Convert the answer tokens to unique id
        answer_token_to_idx = build_vocab([answer for answer in answers])

        # convert the tokens in all questions to unique id
        question_token_to_idx = build_vocab(
            [question for question in questions],
            min_token_count=args.unk_threshold,
            punct_to_keep=[';', ','],
            punct_to_remove=['?', '.']
        )
        
        answer_idx_to_token = {}
        question_idx_to_token = {}

        # create a reverse dictionary for answer idx to token mapping
        for key, value in answer_token_to_idx.items():
            answer_idx_to_token[value] = key

        # create a reverse dictionary for question idx to token mapping
        for key, value in question_token_to_idx.items():
            question_idx_to_token[value] = key

        # dump all the dictionaries as a single JSON file
        vocab = {
            "question_token_to_idx": question_token_to_idx,
            "answer_token_to_idx": answer_token_to_idx,
            "question_idx_to_token": question_idx_to_token,
            "answer_idx_to_token": answer_idx_to_token
        }

    if args.input_vocab_json != "":
        print("Loading vocab...")
        if args.expand_vocab == 1:
            new_vocab = vocab
        with open(args.input_vocab_json, 'r') as f:
            vocab = json.load(f)

        if args.expand_vocab == 1:
            num_new_words = 0
            for word in new_vocab["question_token_to_idx"]:
                if word not in vocab["question_token_to_idx"]:
                    print("Found new word %s" % word)
                    idx = len(vocab["question_token_to_idx"])
                    vocab["question_token_to_idx"][word] = idx
                    num_new_words += 1
                print("Found %d new words" % num_new_words)

    if args.output_vocab_json != "":
        with open(args.output_vocab_json, 'w') as f:
            json.dump(vocab, f)

    # Encode all questions and programs
    # This converts question strings to integers
    print("Encoding data")
    questions_encoded = []
    _answers = []

    for question, answer in zip(questions, answers):
        question_tokens = tokenize(question, punct_to_keep=[';', ','],
                                   punct_to_remove=['?', '.'])
        question_encoded = encode(question_tokens,
                                  vocab["question_token_to_idx"],
                                  allow_unk=args.encode_unk == 1)
        questions_encoded.append(question_encoded)
        _answers.append(vocab["answer_token_to_idx"][answer])

    # Pad encoded questions and programs
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab["question_token_to_idx"]["<NULL>"])

    # Create h5 dataset file
    print("Writing output")
    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    print("Questions encoded shape is {}".format(questions_encoded.shape))
    print("Length of answer tokens is {}".format(len(answer_token_to_idx)))

    with h5py.File(args.output_h5_file, 'w') as f:
        f.create_dataset("questions", data=questions_encoded)

        if len(_answers) > 0:
            f.create_dataset("answers", data=np.asarray(_answers))


if __name__ == "__main__":
    args = parser.parse_args()
    if args.preprocess_questions_for == "train":
        args.input_questions = "./data/train/train_questions.txt"
        args.input_answers = "./data/train/train_answer.txt"
        args.output_h5_file = "./output/train_questions.h5"
    elif args.preprocess_questions_for == "test":
        args.input_questions = "./data/test/test_questions.txt"
        args.input_answers = "./data/test/test_answer.txt"
        args.output_h5_file = "./output/test_questions.h5"
        args.input_vocab_json = "./output/vocab.json"
    
    main(args)
