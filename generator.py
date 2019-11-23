import cv2
import os
import numpy as np
import random
import json
import math

train_size = 9800  # size of the training dataset
test_size = 200    # size of the testing dataset
img_size = 75     # image size
size = 5           # minimum size of the objects
nb_questions = 10  # no. of questions per image

dirs = './data'
test_images = './data/test/images'
train_images = './data/train/images'

# each image will contain 6 objects of 6 different colors
colors = [
    (0, 0, 255),      # red
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 156, 255),    # orange
    (128, 128, 128),  # gray
    (0, 255, 255)     # yellow
]

colors_code = ['red', 'green', 'blue', 'orange', 'gray', 'yellow']

# Create the directories for the first time
try:
    os.makedirs(dirs)
except:
    print('directory {} already exists'.format(dirs))

try:
    os.makedirs(test_images, exist_ok=True)
except:
    print('directory {} already exists'.format(test_images))

try:
    os.makedirs(train_images, exist_ok=True)
except:
    print('directory {} already exists'.format(train_images))


def center_generate(objects):
    while True:
        pas = True
        center = np.random.randint(0+size, img_size - size, 2)
        if len(objects) > 0:
            for name, c, shape in objects:
                if ((center - c) ** 2).sum() < ((size * 2) ** 2):
                    pas = False
        if pas:
            return center


def build_dataset(dataset_type, count):
    """
    Method to create dataset for training and testing

    :param dataset_type: the type of the dataset
    :param count: a loop counter
    :return:
    """
    objects = []  # list to contain all objects in the image
    img = np.zeros((img_size, img_size, 3))

    nr_circles = 0  # no of circles in the image
    nr_rectangles = 0  # no of rectangles in the image

    # each image will contain 6 objects of 6 different colors
    # generate each object and allocate a center to it
    for color_id, color in enumerate(colors):
        center = center_generate(objects)  # generate center for the object

        # generate a random number from (0,1)
        # if the number generated is less than 0.5, create a rectangle
        # else create a circle
        if random.random() < 0.5:
            start = (center[0] - size, center[1] - size)  # start vertex
            end = (center[0] + size, center[1] + size)    # end vertex

            cv2.rectangle(img, start, end, color, -1)
            # cv2.imshow('image.jpg', img)
            objects.append((color_id, center, 'rectangle'))

            nr_rectangles += 1
        else:
            center_ = (center[0], center[1])
            cv2.circle(img, center_, size, color, -1)
            # cv2.imshow('image.jpg', img)
            objects.append((color_id, center, 'circle'))
            nr_circles += 1

        # save the images locally either to test or train dataset directory
        if dataset_type == 'test':
            cv2.imwrite(test_images + '/img_' + str(count).zfill(2) + '.jpeg', img)
        else:
            cv2.imwrite(train_images + '/img_' + str(count).zfill(2) + '.jpeg', img)

    questions = []
    answers = []
    answer = None

    # randomly choose any 10 questions from the questions list
    with open('questions.json') as json_file:
        data = json.load(json_file)
        chosen_questions = random.sample(data["questions"], k=10)

    for question in chosen_questions:
        # choose a random color
        # this color would be used to identify an object within the image
        # the questions would then be asked for that specific object
        color = random.randint(0, 5)

        # What is the shape of the <C> colored object?
        if question['id'] == 1:
            question["question"] = question["question"].replace("<C>", colors_code[color])
            answer = objects[color][2]

        # Is the <C> colored object on top of the image?
        if question['id'] == 2:
            question["question"] = question["question"].replace("<C>", colors_code[color])
            if objects[color][1][1] < (img_size / 2):
                answer = True
            else:
                answer = False

        # Is the <C> colored object on bottom of the image?
        if question['id'] == 3:
            question["question"] = question["question"].replace("<C>", colors_code[color])
            if objects[color][1][1] > (img_size / 2):
                answer = True
            else:
                answer = False

        # Is the <C> colored object on left of the image?
        if question['id'] == 4:
            question["question"] = question["question"].replace("<C>", colors_code[color])
            if objects[color][1][0] < (img_size / 2):
                answer = True
            else:
                answer = False

        # Is the <C> colored object on right of the image?
        if question['id'] == 5:
            question["question"] = question["question"].replace("<C>", colors_code[color])
            if objects[color][1][0] > (img_size / 2):
                answer = True
            else:
                answer = False

        # How many objects have similar shape as of the <C> colored object?
        if question['id'] == 6:
            question["question"] = question["question"].replace("<C>", colors_code[color])
            if objects[color][2] == "circle":
                answer = nr_circles
            else:
                answer = nr_rectangles

        # Are there more than one object of similar shape as of <C> colored object?
        if question["id"] == 7:
            question["question"] = question["question"].replace("<C>", colors_code[color])
            if objects[color][2] == "circle" and nr_circles > 1:
                answer = True
            elif objects[color][2] == "rectangle" and nr_rectangles > 1:
                answer = True
            else:
                answer = False

        # Are there more than one square in the image?
        if question["id"] == 8:
            if nr_rectangles > 1:
                answer = True
            else:
                answer = False

        # Are there more than one circle in the image?
        if question["id"] == 9:
            if nr_circles > 1:
                answer = True
            else:
                answer = False

        # How many squares are there in the image?
        if question["id"] == 10:
            answer = nr_rectangles

        # How many circles are there in the image?
        if question["id"] == 11:
            answer = nr_circles

        # What is the color of the nearest object from <C> colored object?
        # What is the shape of the nearest object from <C> colored object?
        # Is the shape of the nearest object similar to the <C> colored object?
        if question["id"] in (12, 14, 16):
            question["question"] = question["question"].replace("<C>", colors_code[color])
            
            min_distance=100000
            for _object in objects:
                distance = math.sqrt(
                    (_object[1][0] - objects[color][1][0]) ** 2 + \
                    (_object[1][1] - objects[color][1][1]) ** 2
                )
                
                if distance < min_distance and distance != 0:
                    min_distance = distance
                    if question["id"] == 12:
                        answer = colors_code[_object[0]]
                        
                    if question["id"] == 14:
                        answer = _object[2]

                    if question["id"] == 16:
                        if _object[2] == objects[color][2]:
                            answer = True
                        else:
                            answer = False

        # What is the color of the farthest object from <C> colored object?
        # What is the shape of the farthest object from <C> colored object?
        # Is the shape of the farthest object similar to the <C> colored object?
        if question["id"] in (13, 15, 17):
            question["question"] = question["question"].replace("<C>", colors_code[color])
            max_distance = 0

            for _object in objects:
                distance = math.sqrt(
                    (_object[1][0] - objects[color][1][0]) ** 2 + \
                    (_object[1][1] - objects[color][1][1]) ** 2
                )
                
                if distance > max_distance:
                    max_distance = distance
                    if question["id"] == 13:
                        answer = colors_code[_object[0]]
                    
                    if question["id"] == 15:
                        answer = _object[2]
                        
                    if question["id"] == 17:
                        if _object[2] == objects[color][2]:
                            answer = True
                        else:
                            answer = False

        # Are there any objects on top of the <C> colored object?
        # Are there any objects on bottom of the <C> colored object?
        # Are there any objects on left of the <C> colored object?
        # Are there any objects on right of the <C> colored object?
        if question["id"] in (18, 19, 20, 21):
            question["question"] = question["question"].replace("<C>", colors_code[color])
            
            for _object in objects:
                if question["id"] == 18:
                    if _object[1][1] < objects[color][1][1]:
                        answer = True
                        break
                    else:
                        answer = False
                
                if question["id"] == 19:
                    if _object[1][1] > objects[color][1][1]:
                        answer = True
                        break
                    else:
                        answer = False
                        
                if question["id"] == 20:
                    if _object[1][0] < objects[color][1][0]:
                        answer = True
                        break
                    else:
                        answer = False
                        
                if question["id"] == 21:
                    if _object[1][0] > objects[color][1][0]:
                        answer = True
                        break
                    else:
                        answer = False

        questions.append(question["question"])
        answers.append(answer)

    if dataset_type == 'test':
        questions_file = "./data/test/test_questions.txt"
        answers_file = "./data/test/test_answer.txt"
    else:
        questions_file = "./data/train/train_questions.txt"
        answers_file = "./data/train/train_answer.txt"

    with open(questions_file, 'a') as f:
        for question in questions:
            f.write(question)
            f.write('\n')

    with open(answers_file, 'a') as f:
        for answer in answers:
            f.write(str(answer))
            f.write('\n')


print('building testing dataset...')
for count in range(test_size):
    build_dataset('test', count)

print('building training dataset...')
for count in range(train_size):
    build_dataset('train', count)
