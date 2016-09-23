# -*- coding: utf-8 -*-
import io
import os
import logging
import pickle
import shutil
import codecs
import numpy as np

from sklearn.utils import validation
from sklearn.datasets import get_data_home
from sklearn.datasets import base

# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

class WixCache:
    data = []
    description = None
    filenames = None
    target = None
    target_names = None

    def __init__(self):
        f = []
        self.filenames = np.array(f)
        t = []
        self.target = np.array(t)

class WixInstance:

    _features = []

    def __init__(self, wix_csv=None):
        if wix_csv is not None:
            segments = str(wix_csv).split(',')
            for segment in segments:
                self._features.append(segment.strip())

    def get_features(self):
        return self._features

    def get_flag_index(self):
        return len(self._features) - 1


class WixDataAdapter:

    logger = logging.getLogger(__name__)
    cache_name = "cache.pkz"
    train_folder = "train"
    test_folder = 'test'
    data_home_path = ''
    cache_path = ''
    wix_path = ''
    #column_index = ''
    category_map = \
        {
            'flag': 12
        }

    def __init__(self, p_category):

        #self.column_index = self.category_map[p_category]
        self.category = p_category
        self.data_home_path = get_data_home()
        self.cache_path = os.path.join(self.data_home_path, self.cache_name)
        self.wix_path = os.path.join(self.data_home_path, 'wix')

    def create_data(self):
        data_home = get_data_home()
        cache_path = os.path.join(data_home, self.cache_name)

        if os.path.exists(cache_path):
            return

        if not os.path.exists(self.wix_path):
            raise Exception('Path not exists: ' + self.wix_path)

        file_paths = []
        # We might have more than a single data source file
        for root, directories, files in os.walk(self.wix_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_paths.append(file_path)
        print (file_paths)

        instances = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                for line in f:
                    instances.append(line)
            f.closed

        counter = 0
        iterator = iter(instances)
        # skip the first line - column names
        next(iterator)
        for row in iterator:
            columns = str(row).split(',')

            train_path = os.path.join(self.wix_path, self.train_folder)
            if not os.path.exists(train_path):
                os.makedirs(train_path)

            category_value = columns[self.column_index].strip()
            category_path = os.path.join(train_path, category_value)

            if not os.path.exists(category_path):
                os.makedirs(category_path)

            file_path = os.path.join(category_path, str(counter) + '.txt')
            with open(file_path, "w") as text_file:
                text_file.write(row)
            counter += 1

    def get_train_data(self, categories, shuffle=True, random_state=42):

        data = None

        # First, check if the cache is built already and load it
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    compressed_content = f.read()
                uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
                cache = pickle.loads(uncompressed_content)
                data = cache['train']
            except Exception as e:
                print('Cache loading failed: ' + e.message)

        # Second, if the cache does not exist, need to create it from plain data source files
        else:
            train_path = os.path.join(self.wix_path, self.train_folder)
            if not os.path.exists(train_path):
                raise Exception('Train path not found: ' + train_path)

            file_paths = []
            for root, directories, files in os.walk(train_path):
                for filename in files:
                    file_path = os.path.join(train_path, filename)
                    file_paths.append(file_path)

            all_lines = []
            for file_path in file_paths:
                with open(file_path, 'rb') as f:

                    lines = f.readlines()
                    iterator = iter(lines)
                    next(iterator)  # skip the first row if we there is a header
                    for line in iterator:
                        all_lines.append(line)

            feature_num = len(all_lines[0].strip().split(',')) -2
            data_table = np.zeros((len(all_lines), feature_num))

            for i in range(len(all_lines)):
                striped = all_lines[i].strip().split(',')
                for j in range(feature_num):
                    if(j>0):
                        data_table[i,j] = float(str(striped[j]))

            cache = dict(train=WixCache())
            cache['train'] = WixCache()
            cache['train'].data = data_table
            data = cache['train']
            data.target_names = categories
            data.description = 'The Wix dataset'

            counter = 0
            target_arr = np.zeros(data_table.shape[0])
            for line in all_lines:
                segments = str(line).split(',')
                last_column = feature_num+1
                instance_value = segments[last_column].strip()
                label_index = data.target_names.index(instance_value)
                target_arr[counter] = label_index
                counter+=1
            data.target = target_arr

            compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')

            with open(self.cache_path, 'wb') as f:
                f.write(compressed_content)



        labels = [(data.target_names.index(cat), cat) for cat in categories]
        # Sort the categories to have the ordering of the labels
        labels.sort()
        labels, categories = zip(*labels)
        mask = np.in1d(data.target, labels)

        # data.filenames = data.filenames[mask]
        data.target = data.target[mask]
        # searchsorted to have continuous labels
        data.target = np.searchsorted(labels, data.target)
        data.target_names = list(categories)
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[mask]


        data.data = data_lst.tolist()

        if shuffle:
            random_state = validation.check_random_state(random_state)
            indices = np.arange(data.target.shape[0])
            random_state.shuffle(indices)
            # data.filenames = data.filenames[indices]
            data.target = data.target[indices]
            # Use an object array to shuffle: avoids memory copy
            data_lst = np.array(data.data, dtype=object)
            data_lst = data_lst[indices]
            data.data = data_lst.tolist()

        return data


    def get_data(self, subset='train', categories=None, shuffle=True, random_state=42):
        cache = None
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    compressed_content = f.read()
                uncompressed_content = codecs.decode(
                    compressed_content, 'zlib_codec')
                cache = pickle.loads(uncompressed_content)
            except Exception as e:
                print(80 * '_')
                print('Cache loading failed')
                print(80 * '_')
                print(e)

        if cache is None:
            cache = self.get_cache(self.wix_path)

        if subset in ('train', 'test'):
            data = cache[subset]
        else:
            raise ValueError(
                "subset can only be 'train', 'test' or 'all', got '%s'" % subset)

        data.description = 'The Wix dataset'

        if categories is not None:
            labels = [(data.target_names.index(cat), cat) for cat in categories]
            # Sort the categories to have the ordering of the labels
            labels.sort()
            labels, categories = zip(*labels)
            mask = np.in1d(data.target, labels)
            data.filenames = data.filenames[mask]
            data.target = data.target[mask]
            # searchsorted to have continuous labels
            data.target = np.searchsorted(labels, data.target)
            data.target_names = list(categories)
            # Use an object array to shuffle: avoids memory copy
            data_lst = np.array(data.data, dtype=object)
            data_lst = data_lst[mask]
            data.data = data_lst.tolist()

        if shuffle:
            random_state = validation.check_random_state(random_state)
            indices = np.arange(data.target.shape[0])
            random_state.shuffle(indices)
            data.filenames = data.filenames[indices]
            data.target = data.target[indices]
            # Use an object array to shuffle: avoids memory copy
            data_lst = np.array(data.data, dtype=object)
            data_lst = data_lst[indices]
            data.data = data_lst.tolist()

        return data

    def get_cache(self, target_path):
        train_path = os.path.join(target_path, self.train_folder)
        test_path = os.path.join(target_path, self.test_folder)

        if not os.path.exists(target_path):
            os.makedirs(target_path)

        if not os.path.exists(train_path):
            os.makedirs(train_path)

        if not os.path.exists(test_path):
            os.makedirs(test_path)

        cache = dict(train=base.load_files(train_path, encoding='utf-8'),
                     test=base.load_files(test_path, encoding='utf-8'))

        # Turn textual instances representation into text the object structure
        instances = list()
        for instance in cache['train'].data:
            instances.append(WixInstance(instance))
        cache['train'].data = instances

        instances = list()
        for instance in cache['test'].data:
            instances.append(WixInstance(instance))
        cache['test'].data = instances

        compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')

        with open(self.cache_path, 'wb') as f:
            f.write(compressed_content)

        shutil.rmtree(target_path)

        return cache

    def get_unclassified_data(self,categories):
        test_path = os.path.join(self.wix_path, self.test_folder)
        if not os.path.exists(test_path):
            raise Exception('Train path not found: ' + test_path)

        file_paths = []
        for root, directories, files in os.walk(test_path):
            for filename in files:
                file_path = os.path.join(test_path, filename)
                file_paths.append(file_path)

        all_lines = []
        for file_path in file_paths:
            with open(file_path, 'rb') as f:

                lines = f.readlines()
                iterator = iter(lines)
                for line in iterator:
                    all_lines.append(line)

        feature_num = len(all_lines[0].strip().split(',')) - 2
        data_table = np.zeros((len(all_lines), feature_num))

        for i in range(len(all_lines)):
            striped = all_lines[i].strip().split(',')
            for j in range(feature_num):
                if (j > 0):
                    data_table[i, j] = float(str(striped[j]))

        cache = dict(test=WixCache())
        cache['test'] = WixCache()
        cache['test'].data = data_table
        data = cache['test']
        data.target_names = categories
        data.description = 'The Wix dataset'

        counter = 0
        target_arr = np.zeros(data_table.shape[0])
        for line in all_lines:
            segments = str(line).split(',')
            last_column = feature_num + 1
            instance_value = segments[last_column].strip()
            label_index = data.target_names.index(instance_value)
            target_arr[counter] = label_index
            counter += 1
        data.target = target_arr

        return data
