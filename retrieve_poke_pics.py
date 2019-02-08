import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


class PokeDataSet:

    def __init__(self, shape=None):
        self.images = []
        self.num_images = 0
        self.pic_dir = './poke_pics/'
        self.batch_idx = 0
        self.pic_shape = shape
        self.pic_paths = self._make_img_paths()

        self._load_and_process_poke_pics()

    def get_poke_pics(self):
        np.random.shuffle(self.images)
        return self.images

    def get_next_poke_batch(self, batch_size=1):
        idx_stop = self.batch_idx + batch_size
        print(idx_stop, self.num_images)
        res = self.images[self.batch_idx:idx_stop]
        self.batch_idx = self.batch_idx + batch_size
        return res

    def _make_img_paths(self):
        return [f"{self.pic_dir}/{pic}" for pic in os.listdir(self.pic_dir)]

    def _process_image(self, img, shape=None):
        img = img[:, :, 0] / 255.0
        if shape is not None:
            img = cv2.resize(img, dsize=shape, interpolation=cv2.INTER_LINEAR)
        img = np.ravel(img)
        return img

    def _load_and_process_poke_pics(self):
        self.images = [self._process_image(
            cv2.imread(pic), self.pic_shape) for pic in self.pic_paths
        ]

        # for pic in os.listdir(self.pic_dir):
        #     img = cv2.imread("{}/{}".format(self.pic_dir, pic))
        #     img = img[:, :, 0] / 255.0
        #     img = cv2.resize(
        #         img, dsize=self.pic_shape, interpolation=cv2.INTER_LINEAR
        #     )
        #     img = np.ravel(img)
        #     print(img)
        #     self.images.append(img)

        self.images = np.asarray(self.images)
        self.num_images = len(self.images)


if __name__ == "__main__":

    def test_images_attribute(poke_obj):
        img = poke_obj.images[0].reshape(poke_obj.pic_shape)
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.show()

    def test_get_pics(poke_obj):
        pics = poke_obj.get_poke_pics()
        img = pics[0].reshape(poke_obj.pic_shape)
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.show()

    def test_get_batch(poke_obj):
        b = poke_obj.get_next_poke_batch(50)
        while len(b) != 0:
            img = b[0].reshape(poke_obj.pic_shape)
            plt.imshow(img, cmap=plt.get_cmap('gray'))
            plt.show()
            b = poke_obj.get_next_poke_batch(50)

    poke_db = PokeDataSet(shape=(32, 32))
    # test_images_attribute(poke_db)
    test_get_pics(poke_db)
    # test_get_batch(poke_db)
