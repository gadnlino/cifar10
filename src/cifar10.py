import os
import pickle

DATASET_FOLDER = './files/dataset'

TRAINING_FILES = [
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5'
]

TEST_FILES = [
    'test_batch'
]

class Cifar10:
    def __init__(self) -> None:
        self.__model = None
        self.__label_meta = None
        self.__images_batches = []

    def __download_files(self):
        import download_files

    def __read_file(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            return dict
    
    def __add_batch(self, dic):
        # Example of an image in a batch
        # {
        #     'label': 1,
        #     'colors': {
        #         'r': [],
        #         'g': [],
        #         'b': []
        #     }
        # }

        batch = []
        cnt = 0

        for data in dic[b'data']:
            img = {
                'label': dic[b'labels'][cnt],
                'colors_original': data,
                'colors': {
                    'r': list(data[0:1024]),
                    'g': list(data[1024:2048]),
                    'b': list(data[2048:3072]),
                }
            }

            batch.append(img)

            cnt += 1

        self.__images_batches.append(batch)

    def run(self):
        self.__download_files()

        for file in TRAINING_FILES:
            batch = self.__read_file(os.path.join(DATASET_FOLDER, file))
            self.__add_batch(batch)
        
        print(len(self.__images_batches))
        print(len(self.__images_batches[0][0]['colors']['r']))
        print(len(self.__images_batches[0][0]['colors']['g']))
        print(len(self.__images_batches[0][0]['colors']['b']))