import cv2
import os
import matplotlib.pyplot as plt


class PokeDataSet:

    def __init__(self):
        self.result = []
        self.res_length = 0
        self.pic_dir = './poke_pics/'

        self._load_all_poke_pics()

    def get_next_poke_batch(self, batch_size=1):
        for ndx in range(0, self.res_length, batch_size):
            yield self.result[ndx:min(ndx + batch_size, self.res_length)]

    def get_poke_pics(self):
        return self.result

    def _load_all_poke_pics(self):
        for pic in os.listdir(self.pic_dir):
            print(pic)
            img_rgb = cv2.imread("{}/{}".format(self.pic_dir, pic))
            img = cv2.cvtColor(img_rgb, cv2.IMREAD_COLOR)
            self.result.append(img[:, :, 0])
        self.res_length = len(self.result)

"""
poke_db = PokeDataSet()
plt.imshow(poke_db.result[0], cmap=plt.get_cmap('gray'))
plt.show()

for b in poke_db.get_next_poke_batch(3):
    plt.imshow(b, cmap=plt.get_cmap('gray'))
    plt.show()
"""