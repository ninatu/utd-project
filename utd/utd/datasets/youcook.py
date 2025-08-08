import os
import pandas as pd
import json
from utd.utd.datasets.base_dataset import BaseDataset
from utd.utd.datasets.video_utils import read_frames


class YouCookDataset(BaseDataset):
    def load_metadata(self):
        with open(os.path.join(self.root, 'youcookii_annotations_trainval.json'), 'rb') as fin:
            all_data = json.load(fin)['database']

        with open(os.path.join(self.root, 'splits', f'{self.split}_list.txt'), 'r') as fin:
            video_ids = fin.readlines()
            video_ids = [os.path.basename(name.strip()) for name in video_ids if name.strip() != '']

            data = []
            for video_id in video_ids:
                for clip_data in all_data[video_id]['annotations']:
                    clip_id = f'{video_id}_{clip_data["id"]}'
                    clip_data['video_id'] = video_id
                    clip_data['clip_id'] = clip_id
                    clip_data['recipe_type'] = all_data[video_id]['recipe_type']
                    data.append(clip_data)

        metadata = pd.DataFrame(data)

        def _get_video_path(sample):
            if self.split == 'train':
                folder = 'training'
            elif self.split == 'val':
                folder = 'validation'
            else:
                raise NotImplementedError
            rel_path = os.path.join(folder, sample['recipe_type'], sample['video_id'], sample['video_id'])
            if os.path.exists(os.path.join(self.root, rel_path + '.mp4')):
                rel_path = rel_path + '.mp4'
            elif os.path.exists(os.path.join(self.root, rel_path + '.mkv')) :
                rel_path = rel_path + '.mkv'
            else:
                rel_path = None
            return rel_path

        metadata['full_path'] = metadata.apply(_get_video_path, axis=1)
        avail_metadata = metadata[metadata['full_path'].notnull()]
        if len(avail_metadata) != len(metadata):
            print(f"{len(metadata) - len(avail_metadata)} videos are missing!")
            metadata = avail_metadata

        metadata['caption'] = metadata['sentence']
        metadata['main_video_id'] = metadata['video_id']
        metadata['video_id'] = metadata['clip_id']
        del metadata['sentence']
        return metadata

    def __getitem__(self, idx):
        item = self.metadata.iloc[idx]
        video_path = os.path.join(self.root, item['full_path'])
        start, end = item['segment']
        video = read_frames(video_path, self.num_frames, start=start, end=end, safe_read=self.safe_read)
        if self.transform is not None:
            video = self.transform(video)

        output = item.to_dict()
        output['video'] = video

        return output