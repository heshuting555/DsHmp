###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2024
###########################################################################
import json
import os
from collections import defaultdict

for split in ['train', 'valid_u']:
    base_dir = 'datasets/mevis/{}'.format(split)
    mevis_json = json.load(open(os.path.join(base_dir, 'meta_expressions.json')))['videos']
    videos = list(mevis_json.keys())
    idx = 0
    refer_id_num = 0
    refer_id_dict = {}
    idx2refer = {}
    refer2idx = defaultdict(list)
    idx2vid = {}
    vid2idx = defaultdict(list)
    for vid in videos:
        vid_data = mevis_json[vid]
        vid_frames = sorted(vid_data['frames'])
        vid_len = len(vid_frames)
        for exp_id, exp_dict in vid_data['expressions'].items():
            meta = {}
            meta['video'] = vid
            obj_id = ''.join([str(x) for x in exp_dict['obj_id']])
            refer_id = vid + '_' + obj_id
            if refer_id not in refer_id_dict:
                refer_id_dict[refer_id] = refer_id_num
                refer_id_num += 1
            exp_dict['idx'] = idx
            idx2refer[idx] = refer_id_dict[refer_id]
            refer2idx[refer_id_dict[refer_id]].append(idx)
            idx2vid[idx] = vid
            vid2idx[vid].append(idx)
            idx += 1

    idx2positive = defaultdict(list)
    idx2negative = defaultdict(list)
    for idx, refer in idx2refer.items():
        all_positive = refer2idx[refer]
        all_negative = vid2idx[idx2vid[idx]]
        idx2negative[idx] = [x for x in all_negative if x not in all_positive]
        idx2positive[idx] = all_positive

    with open(os.path.join(base_dir, 'idx2refer.json'), 'w') as outfile:
        json.dump(idx2refer, outfile, indent=4)
    with open(os.path.join(base_dir, 'idx2positive.json'), 'w') as outfile:
        json.dump(idx2positive, outfile, indent=4)
    with open(os.path.join(base_dir, 'idx2negative.json'), 'w') as outfile:
        json.dump(idx2negative, outfile, indent=4)

    ab = {}
    ab['videos'] = mevis_json
    with open(os.path.join(base_dir, 'meta_expressions_idx.json'), 'w') as outfile:
        json.dump(ab, outfile, indent=4)
    print(split, 'split')
