import os
import argparse
import json
import h5py
import cv2
import numpy as np
import tensorflow as tf

from util import log
from tensorflow.data import Dataset
from tensorflow.keras.utils import to_categorical


class EvalManager:
    def __init__(self):
        # collection of batches (not flattened)
        self._ids = []
        self._predictions = []
        self._groundtruths = []

    def add_batch(self, id, prediction, groundtruth):
        # for now, store them all (as a list of mini batch chunks)
        self._ids.append(id)
        self._predictions.append(prediction)
        self._groundtruths.append(groundtruth)

    def report(self):
        # report L2 loss
        log.info("Computing scores...")
        correct_prediction_nr = 0
        count_nr = 0
        correct_prediction_r = 0
        count_r = 0

        for id, pred, gt in zip(self._ids, self._predictions,
                                self._groundtruths):
            for i in range(pred.shape[0]):
                # relational
                if np.argmax(gt[i, :]) < NUM_COLOR:
                    count_r += 1
                    if np.argmax(pred[i, :]) == np.argmax(gt[i, :]):
                        correct_prediction_r += 1
                # non-relational
                else:
                    count_nr += 1
                    if np.argmax(pred[i, :]) == np.argmax(gt[i, :]):
                        correct_prediction_nr += 1

        avg_nr = float(correct_prediction_nr) / count_nr
        log.infov("Average accuracy of non-relational questions: {}%".format(
            avg_nr * 100))
        avg_r = float(correct_prediction_r) / count_r
        log.infov(
            "Average accuracy of relational questions: {}%".format(avg_r * 100))
        avg = float(correct_prediction_r + correct_prediction_nr) / (
                    count_r + count_nr)
        log.infov("Average accuracy: {}%".format(avg * 100))


class Evaler:

    @staticmethod
    def get_model_class(model_name):
        if model_name == 'baseline':
            from model_baseline import Model
        elif model_name == 'rn':
            from model_rn import Model
        else:
            raise ValueError(model_name)
        return Model

    def __init__(self, config):
        self.config = config
        self.train_dir = config.train_dir
        log.info("self.train_dir = %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        # --- create model ---
        Model = self.get_model_class(config.model)
        log.infov('Using Model class: %s', Model)
        self.model = Model(config)

        tf.compat.v1.reset_default_graph()
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.step_op = tf.no_op(name='step_no_op')

        tf.set_random_seed(120)

        self.session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 0},
        )

        self.session = tf.Session(config=self.session_config)

        # --- checkpoint and monitoring ---
        self.saver = tf.train.Saver(max_to_keep=5)

        self.checkpoint_path = config.checkpoint_path
        if self.checkpoint_path is None and self.train_dir:
            self.checkpoint_path = tf.train.latest_checkpoint(self.train_dir)
        if self.checkpoint_path is None:
            log.warn("No checkpoint is given. Just random initialization :-)")
            self.session.run(tf.global_variables_initializer())
        else:
            log.info("Checkpoint path : %s", self.checkpoint_path)

    def eval_run(self, questions, answers, images):
        # load checkpoint
        if self.checkpoint_path:
            self.saver.restore(self.session, self.checkpoint_path)
            log.info('Loaded from checkpoint!')

        log.infov('Start 1-epoch Inference and Evaluation')

        max_steps = (images.shape[0] // self.batch_size) + 1
        log.info('max steps = %d', max_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.session, coord=coord,
                                               start=True)

        evaler = EvalManager()

        dataset_img = Dataset.from_tensor_slices((images)).batch(
            self.config.batch_size
        )
        dataset_ques = Dataset.from_tensor_slices((questions)).batch(
            self.config.batch_size * 10
        )
        dataset_ans = Dataset.from_tensor_slices((answers)).batch(
            self.config.batch_size * 10
        )

        # Create iterators to iterate over the different batches
        iterator_img = tf.compat.v1.data.make_initializable_iterator(
            dataset_img
        )
        iterator_ques = tf.compat.v1.data.make_initializable_iterator(
            dataset_ques
        )
        iterator_ans = tf.compat.v1.data.make_initializable_iterator(
            dataset_ans
        )

        # Create an operation to get the next batch
        next_image_batch = iterator_img.get_next()
        next_question_batch = iterator_ques.get_next()
        next_answer_batch = iterator_ans.get_next()

        self.session.run(tf.compat.v1.global_variables_initializer())

        # Initialize all the batch iterators
        self.session.run(iterator_img.initializer)
        self.session.run(iterator_ques.initializer)
        self.session.run(iterator_ans.initializer)
        try:
            for s in range(max_steps):
                batch_images = []

                # fetch the next image batch
                img = self.session.run(next_image_batch)

                # Copy each image 10 times. Thus the batch to test
                # would have same size at dimension 0
                for j in range(img.shape[0]):
                    for _ in range(10):
                        batch_images.append(img[j])
                batch_images = np.asarray(batch_images)
                batch_ques = self.session.run(next_question_batch)
                batch_ans = self.session.run(next_answer_batch)
                batch_ans = to_categorical(
                    batch_ans,
                    num_classes=self.config.ans_dim,
                    dtype=int
                )
                step, loss, prediction_pred, prediction_gt = \
                    self.run_single_step(batch_images, batch_ques, batch_ans)
                self.log_step_message(s, loss)
                evaler.add_batch(batch_chunk['id'], prediction_pred, prediction_gt)
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        try:
            coord.join(threads, stop_grace_period_secs=3)
        except RuntimeError as e:
            log.warning(str(e))

        evaler.report()
        log.infov('Evaluation complete.')

    def run_single_step(self, images, ques, ans):
        [step, accuracy, all_preds, all_targets, _] = self.session.run(
            [self.global_step, self.model.accuracy, self.model.all_preds,
             self.model.ans, self.step_op],
            feed_dict={
                self.model.img: images,
                self.model.q: ques,
                self.model.ans: ans
            }
        )

        return step, accuracy, all_preds, all_targets

    def log_step_message(self, step, accuracy, is_train=False):
        log_fn = log.info
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "batch total-accuracy (test): {test_accuracy:.2f}% "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         test_accuracy=accuracy * 100
                         )
               )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--padding', type=str, default='same')
    parser.add_argument('--model', type=str, default='rn',
                        choices=['rn', 'baseline'])
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--vocab_json', type=str, default='./output/vocab.json')
    parser.add_argument('--test_questions_h5', type=str,
                        default='./output/test_questions.h5')
    parser.add_argument('--test_images_path', type=str,
                        default='./data/test/images')
    config = parser.parse_args()

    log.infov('Loading the vocabulary...')
    with open(config.vocab_json, 'r') as f:
        vocab = json.load(f)

    # load all the questions and answers for the images
    with h5py.File(config.test_questions_h5, 'r') as f:
        log.infov('Loading the questions...')
        questions = f['questions'][()]

        log.infov('Loading the answers...')
        answers = f['answers'][()]

    test_images = []  # list to contain all the testing images
    log.infov('Loading training images...')
    for image in sorted(os.listdir(config.test_images_path)):
        image = cv2.imread(os.path.join(config.test_images_path, image),
                           cv2.IMREAD_COLOR)
        image = image / 255.  # normalize images
        test_images.append(image)
    test_images = np.asarray(test_images)  # convert the list to numpy array
    log.warning('All images have been successfully loaded.')
    log.infov('{} images were loaded.'.format(test_images.shape[0]))

    config.question_token_to_idx = vocab['question_token_to_idx']
    config.ques_dim = questions.shape[1]
    config.ans_dim = len(vocab['answer_token_to_idx'])

    evaler = Evaler(config)
    evaler.eval_run(questions, answers, test_images)


if __name__ == '__main__':
    main()