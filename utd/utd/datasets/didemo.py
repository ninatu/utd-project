import os
import pandas as pd
import json
import numpy as np
from utd.utd.datasets.base_dataset import BaseDataset


class DIDEMODataset(BaseDataset):
    def load_metadata(self):
        assert self.split in ["train", "val", "test"]
        video_json_path = os.path.join(self.root, 'LocalizingMoments', 'data', f"{self.split}_data.json")

        with open(video_json_path, 'r') as fp:
            meta_videos = json.load(fp)
        video_ids = np.unique([itm["video"] for itm in meta_videos]).tolist()

        caption_dict = {}

        for itm in meta_videos:
            description = itm["description"]
            times = itm["times"]
            video = itm["video"]

            # each video is split into 5-second temporal chunks
            # average the points from each annotator
            start_ = np.mean([t_[0] for t_ in times]) * 5
            end_ = (np.mean([t_[1] for t_ in times]) + 1) * 5
            if video in caption_dict:
                caption_dict[video]["start"].append(start_)
                caption_dict[video]["end"].append(end_)
                caption_dict[video]["text"].append(description)
            else:
                caption_dict[video] = {}
                caption_dict[video]["start"] = [start_]
                caption_dict[video]["end"] = [end_]
                caption_dict[video]["text"] = [description]

        metadata = []
        features_path = os.path.join(self.root, 'videos')
        broken_ids = [
            '12090392@N02_13482799053_87ef417396.mov'
        ]
        for cur_root, dub_dir, video_files in os.walk(features_path):
            for video_file in video_files:
                video_id = os.path.splitext(video_file)[0]
                if video_id not in video_ids:
                    continue
                if video_id in broken_ids:
                    continue
                file_path = os.path.join(cur_root, video_file)
                file_path = os.path.relpath(file_path, self.root)
                metadata.append({
                    'video_id': video_id,
                    'caption': " ".join(caption_dict[video_id]["text"]),
                    'full_path': file_path,
                })
        if len(metadata) != len(video_ids):
            print(f"{len(video_ids) - len(metadata)} videos are missing!")

        metadata = pd.DataFrame(metadata)
        return metadata
