import os
import pandas as pd
from utd.utd.datasets.base_dataset import BaseDataset


class LSMDCDataset(BaseDataset):
    def load_metadata(self):
        assert self.split in ["train", "val", "test"]
        video_json_path_dict = {}
        if self.split == 'train':
            video_json_path_dict["train"] = os.path.join(self.root, 'lsmdc2016', "LSMDC16_annos_training.csv")
        elif self.split == 'val':
            video_json_path_dict["val"] = os.path.join(self.root, 'lsmdc2016', "LSMDC16_annos_val.csv")
        else:
            video_json_path_dict["test"] = os.path.join(self.root, 'lsmdc2016', "LSMDC16_challenge_1000_publictect.csv")

        # <CLIP_ID>\t<START_ALIGNED>\t<END_ALIGNED>\t<START_EXTRACTED>\t<END_EXTRACTED>\t<SENTENCE>
        # <CLIP_ID> is not a unique identifier, i.e. the same <CLIP_ID> can be associated with multiple sentences.
        # However, LSMDC16_challenge_1000_publictect.csv has no repeat instances
        video_id_list = []
        caption_dict = {}
        with open(video_json_path_dict[self.split], 'r') as fp:
            for line in fp:
                line = line.strip()
                line_split = line.split("\t")
                assert len(line_split) == 6
                clip_id, start_aligned, end_aligned, start_extracted, end_extracted, sentence = line_split
                caption_dict[clip_id] = {
                    'start': start_aligned,
                    'end': end_aligned,
                    'text': sentence,
                    'clip_id': clip_id
                }
                if clip_id not in video_id_list: video_id_list.append(clip_id)

        video_dict = {}
        for features_path in [
            os.path.join(self.root, 'avi'),
            os.path.join(self.root, 'avi-m-vad-aligned')
        ]:
            for cur_root, dub_dir, video_files in os.walk(features_path):
                for video_file in video_files:
                    video_id = ".".join(video_file.split(".")[:-1])
                    if video_id not in video_id_list:
                        continue
                    file_path = os.path.join(cur_root, video_file)
                    file_path = os.path.relpath(file_path, self.root)
                    video_dict[video_id] = file_path

        videos_ids = list(sorted(caption_dict.keys()))
        videos_ids = [video_id for video_id in videos_ids if video_id in video_dict]

        if len(videos_ids) != len(video_id_list):
            print(f"{len(video_id_list) - len(videos_ids)} videos are missing!")

        metadata = pd.DataFrame({'video_id': videos_ids})
        metadata['full_path'] = metadata.apply(lambda x: video_dict[x.video_id], axis=1)
        metadata['caption'] = metadata.apply(lambda x: caption_dict[x.video_id]['text'], axis=1)
        metadata['start'] = metadata.apply(lambda x: caption_dict[x.video_id]['start'], axis=1)
        metadata['end'] = metadata.apply(lambda x: caption_dict[x.video_id]['end'], axis=1)

        return metadata
