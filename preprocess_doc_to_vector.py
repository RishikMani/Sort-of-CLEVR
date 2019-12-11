import numpy as np
import h5py
import json
import argparse

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

parser = argparse.ArgumentParser()
parser.add_argument("--preprocess_questions_for", default="train", type=str,
                    help="Please provide in the type of dataset for which you"
                         "want to process questions. The allowed datasets are"
                         "train and test.")
parser.add_argument("--input_questions", default=None, type=str,
                    help="The path to the input questions file")
parser.add_argument("--input_vocab_json", default="")
parser.add_argument("--output_h5_file", default=None, type=str,
                    help="Save path for the question and answer features")
parser.add_argument("--output_vocab_json", default="./output/vocab.json",
                    help="Path where the vocabulary from questions and answers "
                         "would be written")
parser.add_argument("--model", default="./output/doc2vec.model",
                    help="Path where the doc2vec model would be written")

output_dir = "./output"
try:
    os.makedirs(output_dir)
except:
    print("directory {} already exists".format(output_dir))


def main(args):
    with open(args.output_vocab_json, 'r') as f:
        vocab = json.load(f)

    with open(args.input_questions, 'r') as f:
        questions = f.read()
    questions = questions.split('\n')
    questions = questions[:-1]  # remove white space at the end
    questions = [question.lower().replace('?', '') for question in questions]

    with open(args.input_answers, 'r') as f:
        answers = f.read()
    answers = answers.split('\n')
    answers = answers[:-1]

    tagged_data = [
        TaggedDocument(words=word_tokenize(question), tags=[str(i)]) \
        for i, question in enumerate(questions)
    ]

    max_epoch = 50
    vec_size = 20
    alpha = 0.025
    model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=0.00025,
                    min_count=1, dm=1)
    model.build_vocab(tagged_data)

    for epoch in range(max_epoch):
        model.train(tagged_data, total_examples=model.corpus_count,
                    epochs=model.iter)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    model.save(args.model)

    questions_encoded = []
    for question in questions:
        question = word_tokenize(question)
        question_vec = model.infer_vector(question)
        questions_encoded.append(question_vec)
    questions_encoded = np.array(questions_encoded)

    answer_encoded = []
    for answer in answers:
        answer_encoded.append(vocab["answer_token_to_idx"][answer])
    answer_encoded = np.array(answer_encoded)

    with h5py.File(args.output_h5_file, 'w') as f:
        f.create_dataset("questions", data=questions_encoded)
        f.create_dataset("answers", data=answer_encoded)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.preprocess_questions_for == "train":
        args.input_questions = "./data/train/train_questions.txt"
        args.input_answers = "./data/train/train_answer.txt"
        args.output_h5_file = "./output/train_doc2vec_features.h5"
    elif args.preprocess_questions_for == "test":
        args.input_questions = "./data/test/test_questions.txt"
        args.input_answers = "./data/test/test_answer.txt"
        args.output_h5_file = "./output/test_doc2vec_features.h5"
        args.input_vocab_json = "./output/vocab.json"

    main(args)
