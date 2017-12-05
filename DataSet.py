import os
import skimage
import skimage.io
import skimage.transform
import csv
import skipthoughts
import random
import numpy as np


class DataSet(object):

    def __init__(self, imagepath, tagfile, image_size):

        self._images = []
        self._tags = []
        self._tag_vecs = []

        with open(tagfile, 'r') as f:
            print('Reading images and tags.')
            reader = csv.reader(f, delimiter=',')

            for idx, line in enumerate(reader):
                tags = line[-1]
                valid_tags = ''

                for tag in tags.split('\t'):
                    text = tag.split(':')[0]
                    if 'hair' in text or 'eye' in text:
                        if not valid_tags:
                            valid_tags += text
                        else:
                            valid_tags += ' ' + text

                if len(valid_tags) > 0: 
                    ID = line[0]
                    filename = os.path.join(imagepath, ID+'.jpg')
                    img = skimage.io.imread(filename)
                    img = skimage.transform.resize(img, (image_size, image_size))
                    self._images.append(img)

                    self._tags.append(valid_tags)

        sent2vec = skipthoughts.load_model()
        self._tag_vecs = skipthoughts.encode(sent2vec, self._tags)

        self._images = np.array(self._images)
        self._tags = np.array(self._tags)
        self._tag_vecs = np.array(self._tag_vecs)

        self._image_num = len(self._tags)
        self._index_in_epoch = 0
        self._N_epoch = 0

        return


    def next_batch(self, batch_size=1):
        
        read_images = []
        wrong_images = []
        vecs = []

        for _ in range(batch_size):

            if self._index_in_epoch >= self._image_num:
                random_idx = np.arange(0, self._image_num)
                np.random.shuffle(random_idx)
                
                self._images = self._images[random_idx]
                self._tags = self._tags[random_idx]
                self._tag_vecs = self._tag_vecs[random_idx]
                
                self._index_in_epoch = 0
                self._N_epoch += 1

            while True:
                random_ID = random.randint(0, self._image_num-1)
                if  self._tags[self._index_in_epoch] not in self._tags[random_ID]:
                    break

            read_images.append(self._images[self._index_in_epoch])
            wrong_images.append(self._images[random_ID])
            vecs.append(self._tag_vecs[self._index_in_epoch])

            self._index_in_epoch += 1

        return read_images, wrong_images, vecs


    @property
    def image_num(self):
        return self._image_num

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def N_epoch(self):
        return self._N_epoch

