import cv2
import tensorflow as tf
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class PokeDatabase(object):

    def __init__(self):
        self.pic_dir = pathlib.Path("./poke_pics/")
        self.pic_db = []
        self.db_size = 0

        self._load_poke_pics()

    def load_and_preprocess_image(self, path):
        image = tf.read_file(path)
        print(path, image)
        return self.preprocess_image(image)

    def preprocess_image(self, image):
        image = tf.image.decode_png(image, channels=1)
        image = tf.image.resize_images(image, [28, 28])
        image = tf.image.convert_image_dtype(image, tf.float32)
        print(image)
        image /= 255.0  # normalize to [0,1] range

        return image

    def _load_poke_pics(self):
        all_image_paths = [str(path) for path in self.pic_dir.iterdir()]
        self.db_size = len(all_image_paths)
        for n in range(3):
            img_path = random.choice(all_image_paths)
            print(img_path)
            img = self.load_and_preprocess_image(img_path)
            print(img)
            img_arr = np.array(img)
            print(img_arr)
            img_arr = np.squeeze(img_arr)
            plt.imshow(img_arr)
            plt.show()


        '''
        for n in range(3):
            image_path = random.choice(all_image_paths)
            img_gs = cv2.imread(image_path, 0)
            cv2.imshow('', img_gs)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        '''

        # Create a list of filenames (for whaledata they are jpegs)
        filenames = ["{}/POKEMON_1.jpeg".format(self.pic_dir)]
        # Create a queue that produces the filenames to read
        filename_queue = tf.train.string_input_producer(filenames)
        pass


pdb = PokeDatabase()