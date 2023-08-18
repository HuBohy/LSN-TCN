from fileinput import filename
import os
import glob
import torch
import random
import numpy as np
import sys
from lipreading.utils import read_txt_lines


class MyDataset(object):
    def __init__(self, modality, data_partition, data_dir, label_fp, annonation_direc=None,
        preprocessing_func=None, data_suffix='.npz'):
        assert os.path.isfile( label_fp ), "File path provided for the labels does not exist. Path input: {}".format(label_fp)
        self._data_partition = data_partition
        self._data_dir = data_dir
        self._data_suffix = data_suffix

        self._label_fp = label_fp
        self._annonation_direc = annonation_direc

        self.fps = 25 if modality == "video" else 16000
        self.is_var_length = True
        self.label_idx = -3

        self.preprocessing_func = preprocessing_func

        self._data_files = []

        self.load_dataset()

    def load_dataset(self):
        self._labels = read_txt_lines(self._label_fp)

        self._get_files_for_partition()

        self.labels_count = {x: 0 for x in self._labels}
        for i, x in enumerate(self._data_files):
            label = self._get_label_from_path( x )
            self.labels_count[label] = self.labels_count[label]+1
            self.list[i] = [ x, self._labels.index( label ) ]

        print('Partition {} loaded'.format(self._data_partition))

    def _get_instance_id_from_path(self, x):
        instance_id = x.split('/')[-1]
        return os.path.splitext( instance_id )[0]

    def _get_label_from_path(self, x):
        return x.split('/')[self.label_idx].split('_')[0]

    def _get_files_for_partition(self):
        dir_fp = self._data_dir
        if not dir_fp:
            return

        search_str_npz = os.path.join(dir_fp, '**', self._data_partition, '*.npz')
        self._data_files.extend(glob.glob( search_str_npz ))

    def load_data(self, filename):
        try:
            if filename.endswith('npz'):
                return np.load(filename, allow_pickle=True)['data']
            else:
                return np.load(filename)
        except IOError:
            print( "Error when reading file: {}".format(filename) )
            sys.exit()

    def _apply_variable_length_aug(self, filename, raw_data):
        try:
            raw_data.shape[0]
        except:
            print(filename, raw_data)
        info = os.path.join(*filename.split('/')[self.label_idx:])
        info = os.path.splitext( info )[0]

        utterance_duration = float( info.split('_')[-1] )/100
        half_interval = int( utterance_duration/2.0 * self.fps)  # num frames of utterance / 2
                
        n_frames = raw_data.shape[0]
        mid_idx = ( n_frames -1 ) // 2  # video has n frames, mid point is (n-1)//2 as count starts with 0
        left_idx = random.randint(0, max(0,mid_idx-half_interval-1)  )   # random.randint(a,b) chooses in [a,b]
        right_idx = random.randint( min( mid_idx+half_interval+1,n_frames ), n_frames  )

        return raw_data[left_idx:right_idx]

    def __getitem__(self, idx):
        raw_data = self.load_data(self.list[idx][0])
        # -- perform variable length on training set
        if ( self._data_partition == 'train' ) and self.is_var_length:
            data = self._apply_variable_length_aug(self.list[idx][0], raw_data)
        else:
            data = raw_data
        preprocess_data = self.preprocessing_func(data)
        label = self.list[idx][1]
        intensity = self.list[idx][0].split('_')[-3]
        return preprocess_data, label, intensity

    def __len__(self):
        return len(self._data_files)


def pad_packed_collate(batch):
    if len(batch) == 1:
        data, lengths, labels_np, intensities = zip(*[(a, a.shape[0], b, c) for (a, b, c) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)]) # a -> a[...,0] for RGB videos
        data = torch.FloatTensor(data)
        lengths = [data.size(1)]

    if len(batch) > 1:
        data_list, lengths, labels_np, intensities = zip(*[(a, a.shape[0], b, c) for (a, b, c) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)]) # a -> a[...,0] for RGB videos
        if data_list[0].ndim == 4:
            max_len, h, w = data_list[0].shape[:3]  # since it is sorted, the longest video is the first one
            data_np = np.zeros(( len(data_list), max_len, h, w))
            data_list = [np.array(x[...,0]) for x in data_list]
        elif data_list[0].ndim == 3:
            max_len, h, w = data_list[0].shape[:3]  # since it is sorted, the longest video is the first one
            data_np = np.zeros(( len(data_list), max_len, h, w))
        elif data_list[0].ndim == 1:
            max_len = data_list[0].shape[0]
            data_np = np.zeros( (len(data_list), max_len))
        for idx in range( len(data_np)):
            data_np[idx][:data_list[idx].shape[0]] = data_list[idx]
        data = torch.FloatTensor(data_np)
    labels = torch.LongTensor(labels_np)
    return data, lengths, labels, intensities
    