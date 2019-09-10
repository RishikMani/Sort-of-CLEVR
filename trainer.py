import os
import json
import cv2
import numpy as np
import h5py
import argparse
import time
import tensorflow as tf

from tensorflow.data import Dataset
from tensorflow.keras.utils import to_categorical

from util import log
from model_rn import Model

parser = argparse.ArgumentParser()

parser.add_argument('--train_question_h5', type=str,
                    default='./output/train_questions.h5')
parser.add_argument('--vocab_json', type=str,
                    default='./output/vocab.json')
parser.add_argument('--train_images_path', type=str,
                    default='./data/train/images')

parser.add_argument('--img_size', type=int, default=128)

parser.add_argument('--rnn_wordvec_dim', type=int, default=300)
parser.add_argument('--rnn_hidden_dim', type=int, default=128)

parser.add_argument('--filters', type=int, default=24)
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--strides', type=int, default=3)
parser.add_argument('--padding', type=str, default='same')

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=2.5e-4)
parser.add_argument('--lr_weight_decay', type=bool, default=False)
parser.add_argument('--output_save_step', type=int, default=10)

parser.add_argument('--model', type=str, default='rn')
parser.add_argument('--checkpoint', default=None)


class Trainer:
    @staticmethod
    def get_model_class(model_name):
        if model_name == 'rn':
            from model_rn import Model
        else:
            raise ValueError(model_name)
        return Model

    def __init__(self, config):
        self.config = config
        self.iterations = config.iterations
        self.output_save_step = config.output_save_step
        
        # hyper_parameter_str = 'Sort-of-CLEVR' + '_lr_' + str(config.learning_rate)
        hyper_parameter_str = 'Sort-of-CLEVR'
        self.train_dir = './train_dir/%s-%s-%s' % (
            config.model,
            hyper_parameter_str,
            time.strftime("%Y%m%d-%H%M%S")
        )
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        log.infov("Train Dir: %s", self.train_dir)

        # Clears the default graph stack and resets the global default
        # graph.
        tf.compat.v1.reset_default_graph()
        
        # Create the model
        self.model = Model(config)
        
        # Returns and create (if necessary) the global step tensor.
        # global_step refers to the number of batches seen by the graph
        # https://bit.ly/2AAqjs1 offers good explanation for global step
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        
        self.learning_rate = config.learning_rate
        if config.lr_weight_decay:
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True,
                name='decaying_learning_rate'
            )

        # Optimizer
        self.optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.learning_rate
            ),
            clip_gradients=20.0,
            name='optimizer_loss'
        )

        # Creates an event file in a given directory and add summaries
        # and events to it.
        self.summary_writer = tf.compat.v1.summary.FileWriter(self.train_dir)

        self.summary_op = tf.compat.v1.summary.merge_all()
        self.plot_summary_op = \
            tf.compat.v1.summary.merge_all(key='plot_summaries')

        # two summary scalars to be written to the events
        # these scalars would be shown on the tensorboard
        self.acc_summary = tf.compat.v1.summary.scalar("training_accuracy",
                                             self.model.accuracy)
        self.loss_summary = tf.compat.v1.summary.scalar("training_loss",
                                              self.model.loss)

        self.checkpoint_secs = 600  # 10 min

        self.session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 0},
        )

        self.saver = tf.compat.v1.train.Saver(max_to_keep=5)
        
    def train(self, questions, answers, images):
        """

        :param questions: list of all the questions
        :param answers: list of all the answers
        :param images: training images
        :return:
        """
        # img_placeholder = tf.placeholder(dtype=tf.float32, shape=[images.shape[0], 128, 128, 3])
        # img = tf.get_variable('img', [images.shape[0], 128, 128, 3])
        # img.assign(img_placeholder)

        # Create dataset tensor slices
        # Remember that for every image we have 10 questions and
        # thus 10 answers. For every image we need to load 10 questions
        # and 10 answers respectively.
        # dataset_img = Dataset.from_tensor_slices((img)).batch(self.config.batch_size)
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
        iterator_img = tf.compat.v1.data.make_initializable_iterator(dataset_img)
        iterator_ques = tf.compat.v1.data.make_initializable_iterator(
            dataset_ques
        )
        iterator_ans = tf.compat.v1.data.make_initializable_iterator(dataset_ans)

        # Create an operation to get the next batch
        next_image_batch = iterator_img.get_next()
        next_question_batch = iterator_ques.get_next()
        next_answer_batch = iterator_ans.get_next()

        with tf.Session(config=self.session_config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            batches = images.shape[0] // self.config.batch_size
            
            log.infov('Training starts')

            for epoch in range(self.iterations):
                log.warning('Epoch {} started'.format(epoch))

                # Initialize all the batch iterators
                sess.run(iterator_img.initializer)
                sess.run(iterator_ques.initializer)
                sess.run(iterator_ans.initializer)

                for batch in range(batches):
                    batch_images = []
                    # epoch_img = sess.run(next_image_batch, feed_dict={img_placeholder: train_images})

                    # fetch the next image batch
                    img = sess.run(next_image_batch)
                    
                    # Copy each image 10 times. Thus the batch to train
                    # would have same size at dimension 0
                    for j in range(img.shape[0]):
                        for _ in range(10):
                            batch_images.append(img[j])

                    batch_images = np.asarray(batch_images)
                    batch_ques = sess.run(next_question_batch)
                    batch_ans = sess.run(next_answer_batch)
                    batch_ans = to_categorical(
                        batch_ans,
                        num_classes=self.config.ans_dim,
                        dtype=int
                    )
                    
                    step, acc, summary, loss, acc_summary, loss_summary =\
                        self.run_single_epoch(
                            sess,
                            batch_images,
                            batch_ques,
                            batch_ans,
                            epoch=epoch
                        )

                    # Add the different summaries to the graph
                    self.summary_writer.add_summary(acc_summary, epoch)
                    self.summary_writer.add_summary(loss_summary, epoch)

                    if step % 10 == 0:
                        self.log_step_message(step, acc, loss)

                if epoch % self.output_save_step == 0:
                    log.infov("Saved checkpoint at %s", self.train_dir)
                    save_path = self.saver.save(
                        sess,
                        os.path.join(self.train_dir, 'model'),
                        global_step=step)
                    
    def run_single_epoch(self, sess, images, questions, answers, epoch):
        """
        Function to run on every epoch to calculate certain values

        :param sess: current tensorflow session
        :param images: batch images
        :param questions: batch questions
        :param answers: batch answers
        :param epoch: epoch number
        :return: global step, accuracy, summary, loss, accuracy and
                 loss tensorboard summaries
        """

        fetch = [self.global_step, self.model.accuracy, self.summary_op,
                 self.model.loss, self.acc_summary, self.loss_summary,
                 self.optimizer]

        try:
            if epoch % self.output_save_step == 0:
                fetch += [self.plot_summary_op]
        except:
            pass

        fetch_values = sess.run(fetch, feed_dict={
                self.model.img: images,
                self.model.ques: questions,
                self.model.ans: answers
            }
        )

        [step, accuracy, summary, loss, acc_summary, loss_summary] = \
            fetch_values[:6]
        
        try:
            if self.plot_summary_op in fetch:
                summary += fetch_values[-1]
        except:
            pass
            
        return step, accuracy, summary, loss, acc_summary, loss_summary
        
    def log_step_message(self, step, accuracy, loss):
        """
        function to show training details at certain time intervals

        :param step: global step
        :param accuracy: training accuracy
        :param loss: training loss
        :return:
        """
        log_fn = log.info
        log_fn((" [step {step:4d}] " +
                "Loss: {loss:.5f} " +
                "Accuracy: {accuracy:.2f}"
                ).format(step=step,
                         loss=loss,
                         accuracy=accuracy * 100
                         )
               )


def main(args):
    """

    :param args: default arguments or given as input
    """
    # load the vocabulary
    log.infov('Loading the vocabulary...')
    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)
        
    # load all the questions and answers for the images
    with h5py.File(args.train_question_h5, 'r') as f:
        log.infov('Loading the questions...')
        questions = f['questions'][()]
        
        log.infov('Loading the answers...')
        answers = f['answers'][()]
    
    train_images = []  # list to contain all the training images
    log.infov('Loading training images...')
    for image in sorted(os.listdir(args.train_images_path)):
        # Images are RGB, so we use cv2.IMREAD_COLOR
        image = cv2.imread(os.path.join(args.train_images_path, image), cv2.IMREAD_COLOR)
        image = image / 255.  # normalize images
        train_images.append(image)
    train_images = np.asarray(train_images)  # convert the list to numpy array
    log.warning('All images have been successfully loaded.')
    log.infov('{} images were loaded.'.format(train_images.shape[0]))
    
    args.question_token_to_idx = vocab['question_token_to_idx']    
    args.ques_dim = questions.shape[1]
    args.ans_dim = len(vocab['answer_token_to_idx'])
    
    trainer = Trainer(args)
    trainer.train(questions, answers, train_images)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
