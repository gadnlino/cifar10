import os
import pickle

DATASET_FOLDER = './files/dataset'

class Cifar10:
    def __init__(self) -> None:
        self.__model = None

    def __download_files(self):
        import download_files

    def __read_file(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            return dict

    def run(self):
        self.__download_files()

        batch1 = self.__read_file(os.path.join(DATASET_FOLDER, 'data_batch_1'))

        print(batch1)
        print(batch1.keys())
        print(type(batch1))