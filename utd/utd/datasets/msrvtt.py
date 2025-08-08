import os
import pandas as pd
import json
import numpy as np
from utd.utd.datasets.base_dataset import BaseDataset


class MSRVTTDataset(BaseDataset):
    def __init__(self, root, split, num_frames, cut='jsfusion', transform=None, part=None, n_parts=None,
                 metadata_root=None, safe_read=None):
        self.cut = cut
        super(MSRVTTDataset, self).__init__(root, split, num_frames, transform, part, n_parts, metadata_root=metadata_root)

    def load_metadata(self):
        json_fp = os.path.join(self.root, 'annotation', 'MSR_VTT.json')
        with open(json_fp, 'r') as fid:
            data = json.load(fid)
        df = pd.DataFrame(data['annotations'])

        split_dir = os.path.join(self.root, 'high-quality', 'structured-symlinks')
        js_test_cap_idx_path = None
        challenge_splits = {"val", "public_server_val", "public_server_test"}
        if self.cut == "miech":
            train_list_path = "train_list_miech.txt"
            test_list_path = "test_list_miech.txt"
        elif self.cut == "jsfusion":
            train_list_path = "train_list_jsfusion.txt"
            test_list_path = "val_list_jsfusion.txt"
            js_test_cap_idx_path = "jsfusion_val_caption_idx.pkl"
        elif self.cut in {"full-val", "full-test"}:
            train_list_path = "train_list_full.txt"
            if self.cut == "full-val":
                test_list_path = "val_list_full.txt"
            else:
                test_list_path = "test_list_full.txt"
        elif self.cut in challenge_splits:
            train_list_path = "train_list.txt"
            if self.cut == "val":
                test_list_path = f"{self.cut}_list.txt"
            else:
                test_list_path = f"{self.cut}.txt"
        else:
            msg = "unrecognised MSRVTT split: {}"
            raise ValueError(msg.format(self.cut))

        train_df = pd.read_csv(os.path.join(split_dir, train_list_path), names=['videoid'])
        test_df = pd.read_csv(os.path.join(split_dir, test_list_path), names=['videoid'])

        if self.split == 'train':
            df = df[df['image_id'].isin(train_df['videoid'])]
        else:
            df = df[df['image_id'].isin(test_df['videoid'])]

        metadata = df.groupby(['image_id'])['caption'].apply(list)

        # use specific caption idx's in jsfusion
        if js_test_cap_idx_path is not None and self.split != 'train':
            caps = pd.Series(np.load(os.path.join(split_dir, js_test_cap_idx_path), allow_pickle=True))
            new_res = pd.DataFrame({'caps': metadata, 'cap_idx': caps})
            new_res['test_caps'] = new_res.apply(lambda x: [x['caps'][x['cap_idx']]], axis=1)
            metadata = new_res['test_caps']

        metadata = pd.DataFrame({'captions': metadata})
        metadata['video_id'] = metadata.index

        def _get_video_path(sample):
            return os.path.join('videos', 'all', sample.name + '.mp4')

        metadata['full_path'] =  metadata.apply(_get_video_path, axis=1)

        return metadata
