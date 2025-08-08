import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from utd.utd.datasets.video_utils import read_frames
from ast import literal_eval


class BaseDataset(Dataset):
    def __init__(self, root, split, num_frames, transform=None, part=None, n_parts=None,
                 metadata_root=None, safe_read=False):
        self.root = root
        self.split = split
        self.num_frames = num_frames
        self.transform = transform
        self.safe_read = safe_read
        self.metadata_root = metadata_root
        self.metadata = self.load_metadata()

        if part is not None:
            step = int(np.ceil(len(self.metadata) / n_parts))
            self.metadata = self.metadata.iloc[step * part: step * (part + 1)]


    def load_metadata(self):
        metadata_root = self.metadata_root
        path = os.path.join(metadata_root, f'{self.split}_info.csv')
        metadata = pd.read_csv(path)

        if 'captions' in metadata:
            metadata['captions'] = metadata['captions'].apply(literal_eval)

        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata.iloc[idx]
        video_path = os.path.join(self.root, item['full_path'])

        if 'segment' in item:
            start, end = literal_eval(item['segment'])
        else:
            start, end = None, None
        video = read_frames(video_path, self.num_frames, start=start, end=end, safe_read=self.safe_read)

        if self.transform is not None:
            video = self.transform(video)

        output = item.to_dict()
        output['video'] = video

        video_id = output['video_id']
        if 'caption' in item:
            output['text'] = [item['caption']]
            output['text_video_id'] = [video_id]
            output['tv_pair_id'] = [video_id]
        elif 'captions' in item:
            output['text'] = item['captions']
            output['text_video_id'] = [video_id] * len(output['text'])
            output['tv_pair_id'] = [f"{video_id}_{i}" for i in range(len(output['text']))]
        else:
            output['text'] = item['class_text']

        return output
