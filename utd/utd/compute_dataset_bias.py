import os.path

import numpy as np
import pandas as pd
from argparse import ArgumentParser
import tqdm
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=["kinetics_400", "kinetics_600", "kinetics_700", "MiT", "ssv2", "ucf", ],
                        help='The dataset, choose from: "kinetics_400", "kinetics_600", "kinetics_700", "MiT", "ssv2", "ucf"')
    parser.add_argument('--train_split', type=str, required=True,
                        help='The dataset split to use, e.g. "train" or "trainlist01" for ucf.')
    parser.add_argument('--test_split', type=str, required=True,
                        help='The dataset split to use, e.g. "test" or "val" or "testlist01" for ucf.')

    parser.add_argument( '--train_embeddings_path', type=str, required=True,
                         help='Path to the .npz file containing embeddings for train_split.')
    parser.add_argument('--test_embeddings_path', type=str, required=True,
                        help='Path to the .npz file containing embeddings for test_split.')

    parser.add_argument('--output_path', help='Path to save the output CSV file containing results for each video.')


    args = parser.parse_args()

    dataset = args.dataset
    train_split = args.train_split
    test_split = args.test_split
    output_path = args.output_path

    print("\n\nDataset:", dataset)
    print("Output path:", output_path)

    train_data_info = pd.read_csv(f'metadata/{dataset}/{train_split}_info.csv')
    train_data_info.index = train_data_info['video_id']
    train_video_ids = train_data_info['video_id'].tolist()

    test_data_info = pd.read_csv(f'metadata/{dataset}/{test_split}_info.csv')
    test_data_info.index = test_data_info['video_id']
    test_video_ids = test_data_info['video_id'].tolist()

    classes_info = pd.read_csv(f'metadata/{dataset}/classes.csv')
    classes_info.index = classes_info['class_name']

    gt_texts = classes_info['class_text'].tolist()
    gt_texts = [[x] for x in gt_texts]

    train_data = np.load(args.train_embeddings_path)['arr_0']
    test_data = np.load(args.test_embeddings_path)['arr_0']

    train_class_name = np.array(train_data_info['class_name'].to_list())
    test_class_name = np.array(test_data_info['class_name'].to_list())

    clf = LogisticRegression(random_state=0)

    clf.fit(train_data, train_class_name)
    scores = clf.predict_log_proba(test_data)

    classes_info_dict = {i: (class_name, classes_info.loc[class_name]['class_text']) for i, class_name in
                         enumerate(clf.classes_)}
    df = []

    for score, (_, item_info) in tqdm.tqdm(zip(scores, test_data_info.iterrows())):
        correct = item_info.class_name
        correct_text = item_info.class_text
        video_id = item_info.video_id

        predict = [classes_info_dict[i][0] for i in (-score).argsort()]
        predict_text = [classes_info_dict[i][1] for i in (-score).argsort()]

        df.append((video_id, correct, correct_text, predict, predict_text))

    df = pd.DataFrame(df, columns=['video_id', 'gt_label', 'gt_label_text',
                                   'predict_labels', 'predict_label_texts'])
    correct = df.apply(axis=1, func=lambda x: x['gt_label'] in x['predict_labels'][:1])
    df['Correct'] = None
    df.loc[correct, 'Correct'] = 'Correct'

    accuracy = (df.apply(axis=1, func=lambda x: x['gt_label'] in x['predict_labels'][:1])).mean()
    accuracy_top5 = (df.apply(axis=1, func=lambda x: x['gt_label'] in x['predict_labels'][:5])).mean()
    accuracy_top10 = (df.apply(axis=1, func=lambda x: x['gt_label'] in x['predict_labels'][:10])).mean()

    print(dataset, 'acc_top1: {:.3f}, acc_top5: {:.3f}, acc_top10: {:.3f}'.format(accuracy, accuracy_top5, accuracy_top10))
    print(dataset, '{:.3f} {:.3f} {:.3f}'.format(accuracy, accuracy_top5, accuracy_top10))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
