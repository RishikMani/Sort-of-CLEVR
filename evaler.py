import os
import argparse
import json
import h5py
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.data import Dataset
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Conv2D
from tensorflow.keras import backend as K

from util import log

parser = argparse.ArgumentParser()
parser.add_argument("--test_questions_h5", type=str,
                    default="./output/test_questions.h5",
                    help="Path to the h5 feature file")
parser.add_argument("--vocab_json", type=str, default="./output/vocab.json",
                    help="Path to the vocabulary")
parser.add_argument("--test_images_path", type=str,
                    default="./data/test/images",
                    help="Path to the testing images")

parser.add_argument("--img_size", type=int, default=75,
                    help="Size of the image")
parser.add_argument("--ques_dim", type=int, default=0,
                    help="Dimension of the questions")
parser.add_argument("--ans_dim", type=int, default=0,
                    help="Dimension of the answers")

parser.add_argument("--rnn_wordvec_dim", type=int, default=300,
                    help="Dimension of the embedding layer")
parser.add_argument("--rnn_hidden_dim", type=int, default=128,
                    help="The output dimension of the RNN")
parser.add_argument("--rnn_dropout", type=int, default=0, help="Dropout ratio")

parser.add_argument("--filters", type=int, default=24,
                    help="Number of filters after the convolution")
parser.add_argument("--kernel_size", type=int, default=5, help="Filter size")
parser.add_argument("--strides", type=int, default=3, help="Stride value")
parser.add_argument("--padding", type=str, default="same",
                    help="Type of padding in convolution")

parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--model", type=str, default="baseline",
                    choices=["rn", "baseline"], help="Type of the model")
parser.add_argument("--checkpoint_path",
                    default="./train_dir/rn-20191126-114955/model-37179",
                    help="Path to the stored checkpoint")


class Evaler:
    @staticmethod
    def get_model_class(model_name):
        if model_name == "rn":
            from model_rn import Model
        elif model_name == "baseline":
            from model_baseline import Model
        else:
            raise ValueError(model_name)
        return Model

    def __init__(self, config):        
        self.config = config
            
        self.batch_size = self.config.batch_size
        tf.compat.v1.set_random_seed(20)
        tf.compat.v1.reset_default_graph()
        
        self.session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True),
            device_count={"GPU": 1},
        )

        self.checkpoint_path = config.checkpoint_path
        self.loader = tf.compat.v1.train.import_meta_graph(
            self.checkpoint_path + ".meta"
        )
        
        self.graph = tf.compat.v1.get_default_graph()
        self.session = tf.compat.v1.Session(config=self.session_config,
                                            graph=self.graph)
        
        log.info("Checkpoint path : %s", self.checkpoint_path)
        
        Model = self.get_model_class(config.model)
        self.model = Model(config)

    def eval_run(self, images, questions, answers):
        log.infov("Start 1-epoch Inference and Evaluation")

        max_steps = (images.shape[0] // self.batch_size)
        log.info("max steps = %d", max_steps)

        # Create datasets for images, questions and answers
        dataset_img = Dataset.from_tensor_slices((images)).batch(
            self.config.batch_size
        )
        dataset_ques = Dataset.from_tensor_slices((questions)).batch(
            self.config.batch_size * 23
        )
        dataset_ans = Dataset.from_tensor_slices((answers)).batch(
            self.config.batch_size * 23
        )

        # Create iterators to iterate over the different datasets created
        iterator_img = tf.compat.v1.data.make_initializable_iterator(
            dataset_img
        )
        iterator_ques = tf.compat.v1.data.make_initializable_iterator(
            dataset_ques
        )
        iterator_ans = tf.compat.v1.data.make_initializable_iterator(
            dataset_ans
        )

        # create an op to fetch the next batch from the different iterators
        next_image_batch = iterator_img.get_next()
        next_question_batch = iterator_ques.get_next()
        next_answer_batch = iterator_ans.get_next()

        # destroy the current TF graph and create a new one
        tf.keras.backend.clear_session()

        with self.session as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            try:
                self.loader.restore(sess, self.checkpoint_path)
                log.info("Model checkpoint has been successfully restored.")
            except Exception as e:
                log.info("Model checkpoint failed to restore.")
                print(e)

            # initialize all the dataset iterators
            sess.run(iterator_img.initializer)
            sess.run(iterator_ques.initializer)
            sess.run(iterator_ans.initializer)

            testing_accuracy = 0

            for _ in range(max_steps):
                # fetch batches to be processed in the current step
                batch_images = sess.run(next_image_batch)
                batch_ques = sess.run(next_question_batch)
                batch_ans = sess.run(next_answer_batch)

                accuracy, predictions, targets = \
                    self.run_single_step(sess, batch_images, batch_ques,
                                         batch_ans)

                testing_accuracy += accuracy

            print(
                "The average testing accuracy is {0}.".format(
                    testing_accuracy / max_steps
                )
            )

    def run_single_step(self, session, images, ques, ans):
        [accuracy, all_preds, all_targets] = session.run(
            [self.model.accuracy, self.model.all_preds, self.model.ans],
            feed_dict={
                'img_1:0': images,
                'ques_1:0': ques,
                'ans_1:0': ans
            }
        )
        for i in range(all_preds.shape[0]):
            print(
                'Target {}, Predicted: {}'.format(
                    vocab['answer_idx_to_token'][str(np.argmax(all_targets[i]))],
                    vocab['answer_idx_to_token'][str(np.argmax(all_preds[i]))]
                    )
                )
        return accuracy, all_preds, all_targets        


if __name__ == "__main__":
    config = parser.parse_args()

    # load the vocabulary in the memory
    log.infov("Loading the vocabulary...")
    with open(config.vocab_json, "r") as f:
        vocab = json.load(f)
        
    # load all the questions and answers for the images into the memory
    with h5py.File(config.test_questions_h5, "r") as f:
        log.infov("Loading the questions...")
        questions = f["questions"][()]

        log.infov("Loading the answers...")
        answers = f["answers"][()]

    # load all the testing images into the memory
    test_images = []  # list to contain all the testing images
    log.infov("Loading testing images...")
    for image in sorted(os.listdir(config.test_images_path)):
        image = cv2.imread(os.path.join(config.test_images_path, image),
                           cv2.IMREAD_COLOR)
        image = image / 255.  # normalize images
        test_images.append(image)
    test_images = np.asarray(test_images)  # convert the list to numpy array
    log.warning("All images have been successfully loaded.")
    log.infov("{} images were loaded.".format(test_images.shape[0]))
    
    config.question_token_to_idx = vocab["question_token_to_idx"]
    config.ques_dim = questions.shape[1]
    config.ans_dim = len(vocab["answer_token_to_idx"])

    answers = to_categorical(
        answers,
        num_classes=config.ans_dim,
        dtype=int
    )
    
    evaler = Evaler(config)
    evaler.eval_run(test_images, questions, answers)
