import json

with open('.\\output\\vocab.json', 'r') as f:
    vocab = json.load(f)


def question2str(question):
    question_text = ''
    for idx in question:
        question_text += vocab['answer_idx_to_token'][str(idx)]
        question_text += ' '
    question_text += '?'
    return question_text


def answer2str(answer, prefix=None):
    answer_text = vocab['answer_idx_to_token'][str(answer)]
    return '[{} Answer: {}]'.format(prefix, answer_text)
