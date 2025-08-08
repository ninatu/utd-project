import os
import pandas as pd
import pickle
from utd.utd.datasets.base_dataset import BaseDataset


class MSVDDataset(BaseDataset):
    def load_metadata(self):
        assert self.split in ["train", "val", "test"]
        video_id_path = os.path.join(self.root, 'msvd_data', f"{self.split}_list.txt")
        caption_file = os.path.join(self.root, 'msvd_data', "raw-captions.pkl")

        with open(video_id_path, 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]

        with open(caption_file, 'rb') as f:
            captions = pickle.load(f)

        metadata = []
        idx = 0
        features_path = os.path.join(self.root, 'YouTubeClips')
        for cur_root, dub_dir, video_files in os.walk(features_path):
            for video_file in video_files:
                video_id = ".".join(video_file.split(".")[:-1])
                if video_id not in video_ids:
                    continue
                file_path = os.path.join(cur_root, video_file)
                file_path = os.path.relpath(file_path, self.root)
                metadata.append({
                    'video_id': video_id,
                    'captions': [' '.join(words) for words in captions[video_id]],
                    'full_path': file_path,
                    'idx': idx,
                })
                idx += 1

        assert len(metadata) == len(video_ids)

        metadata = pd.DataFrame(metadata)
        return metadata

