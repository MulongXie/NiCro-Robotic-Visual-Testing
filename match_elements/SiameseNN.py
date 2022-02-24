import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from numpy.random import randint as randint

import keras
from keras.models import Model, Sequential, load_model
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy


class SiameseModel:
    def __init__(self):
        self.model = None
        self.build_model()

        self.trained_epoch_number = 0  # record how many epochs the model has been trained

    def build_model(self):
        # Building a sequential model
        input_shape = (100, 100, 3)
        left_input = keras.layers.Input(input_shape)
        right_input = keras.layers.Input(input_shape)

        w_init = keras.initializers.RandomNormal(mean=0.0, stddev=1e-2)
        b_init = keras.initializers.RandomNormal(mean=0.5, stddev=1e-2)

        model = keras.models.Sequential([
            keras.layers.Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, kernel_initializer=w_init, bias_initializer=b_init, kernel_regularizer=l2(2e-4)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(128, (7, 7), activation='relu', kernel_initializer=w_init, bias_initializer=b_init, kernel_regularizer=l2(2e-4)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(128, (4, 4), activation='relu', kernel_initializer=w_init, bias_initializer=b_init, kernel_regularizer=l2(2e-4)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(256, (4, 4), activation='relu', kernel_initializer=w_init, bias_initializer=b_init, kernel_regularizer=l2(2e-4)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='sigmoid', kernel_initializer=w_init, bias_initializer=b_init)
        ])
        encoded_l = model(left_input)
        encoded_r = model(right_input)

        subtracted = keras.layers.Subtract()([encoded_l, encoded_r])
        prediction = keras.layers.Dense(1, activation='sigmoid', bias_initializer=b_init)(subtracted)
        siamese_net = Model(input=[left_input, right_input], output=prediction)
        siamese_net.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0006))
        # plot_model(siamese_net, show_shapes=True, show_layer_names=True)
        self.model = siamese_net

    def train_on_batch(self, batch_x, batch_y):
        '''
        :param batch_x: (2, batch_size, 100, 100, 3), image pairs
        :param batch_y: (batch_size, 1), indicate if the pair's classes are same(1) or different(0)
        '''
        loss = self.model.train_on_batch(batch_x, batch_y)
        return loss

    def train(self, epochs_num, data):
        loss_list = []
        accuracy_list = []
        for epoch in range(1, epochs_num):
            batch_x, batch_y = data.get_siamese_data_bath()
            loss = self.train_on_batch(batch_x, batch_y)
            loss_list.append((epoch, loss))
            print('Epoch:', epoch, ', Loss:', loss)
            if epoch % 200 == 0:
                print("=============================================")
                accuracy = data.nway_one_shot(self.model, 9, 100)
                accuracy_list.append((epoch, accuracy))
                print('Accuracy as of', epoch, 'epochs:', accuracy)
                print("=============================================")
                if accuracy > 99:
                    print("Achieved more than 90% Accuracy")
        return loss_list, accuracy_list

    def load_model(self, model_file='E:/Mulong/Datasets/siamese/siamese.h5'):
        self.model = load_model(model_file)
        print('Model loaded from:', model_file)

    def save_model(self, save_path='E:/Mulong/Datasets/siamese/siamese.h5'):
        self.model.save(save_path)
        print('Model save to:', save_path)


class SiameseData:
    def __init__(self, data_dir='E:/Mulong/Datasets/siamese/fruits/Training', no_of_load_images_in_each_class=10, batch_size=64, train_test_split=0.7):
        self.data_dir = data_dir
        self.x = []
        self.y = []

        self.no_of_loaded_images_in_each_class = no_of_load_images_in_each_class
        self.no_of_loaded_classes = None

        self.train_test_split = train_test_split
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

        self.batch_size = batch_size
        self.siamese_x_batch = []  # (2, batch_size, 100, 100, 3), image pairs
        self.siamese_y_batch = []  # (batch_size, 1), indicate if the pair's classes are same(1) or different(0)

    def load_data_classification(self, data_dir='E:/Mulong/Datasets/siamese/fruits/Training', no_of_load_images_in_each_class=10):
        '''
        :return:
            x: (Num of images: Num of classes x no_of_files_in_each_class, 100, 100, 3), the resized image
            y: (Num of images: Num of classes x no_of_files_in_each_class, 1), label of class 0 ~ Num of classes-1
        '''
        if data_dir != self.data_dir: self.data_dir = data_dir
        if no_of_load_images_in_each_class != self.no_of_loaded_images_in_each_class: self.no_of_loaded_images_in_each_class = no_of_load_images_in_each_class

        # Read all the folders in the directory, each folder for one class
        class_folders = os.listdir(data_dir)
        self.no_of_loaded_classes = len(class_folders)
        print(self.no_of_loaded_classes, "classes found in the dataset")

        # Using just few images per category
        self.x = []
        self.y = []
        y_label = 0
        for folder_name in class_folders:
            class_dir = os.path.join(data_dir, folder_name)
            files_list = os.listdir(class_dir)
            for file_name in files_list[:no_of_load_images_in_each_class]:
                self.x.append(np.asarray(Image.open(os.path.join(class_dir, file_name)).convert('RGB').resize((100, 100))))
                self.y.append(y_label)
            y_label += 1
        self.x = np.asarray(self.x) / 255.0
        self.y = np.asarray(self.y)
        print('X, Y shape', self.x.shape, self.y.shape)

    def split_data(self, train_test_split=0.7):
        if train_test_split != self.train_test_split: self.train_test_split = train_test_split

        # Split the dataset based on class
        train_classes_size = int(self.no_of_loaded_classes * train_test_split)
        test_classes_size = self.no_of_loaded_classes - train_classes_size
        print(train_classes_size, 'classes for training and', test_classes_size, ' classes for testing')

        # Training Split
        train_images_size = train_classes_size * self.no_of_loaded_images_in_each_class
        self.x_train = self.x[:train_images_size]
        self.y_train = self.y[:train_images_size]
        # Testing Split
        self.x_test = self.x[train_images_size:]
        self.y_test = self.y[train_images_size:]

        print('X&Y shape of training data :', np.shape(self.x_train), 'and', np.shape(self.y_train))
        print('X&Y shape of testing data :', np.shape(self.x_test), 'and', np.shape(self.y_test))

    def get_siamese_data_bath(self, batch_size=64):
        '''
        Get a batch of image pairs that can be same-class(1) or diff-class(0), which is the required format for Siamese Model
        :returns
            batch_x: (2, batch_size, 100, 100, 3), image pairs
            batch_y: (batch_size, 1), indicate if the pair's classes are same(1) or different(0)
        '''
        if batch_size != self.batch_size: self.batch_size = batch_size

        # initialize a batch of image pairs
        batch_x = [np.zeros((batch_size, 100, 100, 3)), np.zeros((batch_size, 100, 100, 3))]
        # initialize half of the batch as same class and the other half as different class
        batch_y = np.zeros(batch_size)
        batch_y[int(batch_size / 2):] = 1
        np.random.shuffle(batch_y)

        # randomly pick
        class_id_min, class_id_max = min(self.y_train), max(self.y_train)
        for i in range(batch_size):
            # randomly pick a class
            picked_class_0 = randint(class_id_min, class_id_max)
            # randomly pick a image in the picked class
            picked_img_id_0 = picked_class_0 * self.no_of_loaded_images_in_each_class + randint(0, self.no_of_loaded_images_in_each_class - 1)
            batch_x[0][i] = self.x_train[picked_img_id_0]
            # If train_y has 0 pick from the same class, else pick from any other class
            if batch_y[i] == 0:
                # randomly pick a image in the same class
                picked_img_id_1 = picked_class_0 * self.no_of_loaded_images_in_each_class + randint(0, self.no_of_loaded_images_in_each_class - 1)
            else:
                # randomly pick another class
                picked_class_1 = np.random.choice([j for j in range(class_id_min, picked_class_0)] + [j for j in range(picked_class_0 + 1, class_id_max)])
                # randomly pick a image in the picked class
                picked_img_id_1 = picked_class_1 * self.no_of_loaded_images_in_each_class + randint(0, self.no_of_loaded_images_in_each_class - 1)
            batch_x[1][i] = self.x_train[picked_img_id_1]

        self.siamese_x_batch, self.siamese_y_batch = batch_x, batch_y
        # print('Training batch of X&Y shape :', np.shape(self.batch_x), 'and', np.shape(self.batch_y))
        return batch_x, batch_y

    def nway_one_shot(self, model, n_way, no_of_testing_img):
        no_of_train_classes = int(self.no_of_loaded_classes * self.train_test_split)
        testing_classes = randint(no_of_train_classes + 1, self.no_of_loaded_classes - 1, no_of_testing_img)

        n_correct = 0
        for testing_class_0 in testing_classes:
            testing_img_id_0 = testing_class_0 * self.no_of_loaded_images_in_each_class + randint(0, self.no_of_loaded_images_in_each_class - 1)
            testing_img_pairs = [np.zeros((n_way, 100, 100, 3)), np.zeros((n_way, 100, 100, 3))]
            for k in range(n_way):
                testing_img_pairs[0][k] = self.x[testing_img_id_0]
                # the first pair is in the same class, the others are in different classes
                if k == 0:
                    testing_img_id_1 = testing_class_0 * self.no_of_loaded_images_in_each_class + randint(0, self.no_of_loaded_images_in_each_class - 1)
                else:
                    testing_class_1 = np.random.choice([j for j in range(no_of_train_classes + 1, testing_class_0)] + [j for j in range(testing_class_0 + 1, self.no_of_loaded_classes)])
                    testing_img_id_1 = testing_class_1 * self.no_of_loaded_images_in_each_class + randint(0, self.no_of_loaded_images_in_each_class - 1)
                testing_img_pairs[1][k] = self.x[testing_img_id_1]

            result = model.predict(testing_img_pairs)
            result = result.flatten().tolist()
            result_index = result.index(min(result))
            if result_index == 0:
                n_correct = n_correct + 1
        print(n_correct, "correctly classified among", no_of_testing_img)
        accuracy = (n_correct * 100) / no_of_testing_img
        return accuracy
