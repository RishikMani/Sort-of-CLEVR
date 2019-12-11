import numpy as np
import json
import h5py
import argparse

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--preprocess_questions_for", default="train", type=str,
                    help="Please provide in the type of dataset for which you"
                         "want to process questions. The allowed datasets are"
                         "train and test.")
parser.add_argument("--input_questions", default=None, type=str,
                    help="Path to the input questions.")
parser.add_argument("--input_vocab_json", default="",
                    help="Path to look for the originally written vocabulary")
parser.add_argument("--output_h5_file", default=None, type=str,
                    help="Path where the features derived from questions and "
                         "answers would be written")
parser.add_argument("--output_vocab_json", default="./output/vocab.json",
                    help="Path to look for the originally written vocabulary")

output_dir = "./output"
try:
    os.makedirs(output_dir)
except:
    print("directory {} already exists".format(output_dir))


def main(args):
    print("Loading questions for encoding...")
    with open(args.input_questions, 'r') as f:
        question_ids = f.read()
    question_ids = question_ids.split("\n")[:-1]
    question_ids = [int(question_id) for question_id in question_ids]
    question_ids = np.array(question_ids)

    le = LabelEncoder()
    label_encoded = le.fit_transform(question_ids)

    ohe = OneHotEncoder(sparse=False)
    label_encoded = label_encoded.reshape(len(label_encoded), 1)
    hot_encoded = ohe.fit_transform(label_encoded)
    print("All questions have been encoded to one hot encoded vectors.")

    print("Loading answers for encoding...")
    with open(args.input_answers, 'r') as f:
        answers = f.read()
    answers = answers.split("\n")
    answers = answers[:-1]

    with open(args.output_vocab_json) as f:
        vocab = json.load(f)

    answer_encoded = []
    for answer in answers:
        answer_encoded.append(vocab["answer_token_to_idx"][answer])
    answer_encoded = np.array(answer_encoded)
    print("All answers have been successfully encoded...")

    print("Creating dataset...")
    with h5py.File(args.output_h5_file, 'w') as f:
        f.create_dataset("questions", data=hot_encoded)
        f.create_dataset("answers", data=np.asarray(answer_encoded))
    print("Dataset created...")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.preprocess_questions_for == "train":
        args.input_questions = "./data/train/train_question_ids.txt"
        args.input_answers = "./data/train/train_answer.txt"
        args.output_h5_file = "./output/train_questions_ids.h5"
    elif args.preprocess_questions_for == "test":
        args.input_questions = "./data/test/test_question_ids.txt"
        args.input_answers = "./data/test/test_answer.txt"
        args.output_h5_file = "./output/test_questions_ids.h5"
        # args.input_vocab_json = "./output/vocab.json"
    main(args)
