# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_source', default=None)
    parser.add_argument("--local_dir", default="~/data/alfworld")
    args = parser.parse_args()

    data_source = args.data_source

    # Support both new JSONL format {"prompt":..., "response":...}
    # and legacy JSON format [{data:[{prompt,response},...]}]
    train_dataset = []
    if data_source.endswith('.jsonl'):
        with open(data_source, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    train_dataset.append(json.loads(line))
    else:
        with open(data_source, 'r') as f:
            dataset = json.load(f)
        # Handle list-of-trajectories format
        if isinstance(dataset, list) and dataset and isinstance(dataset[0], dict):
            first = dataset[0]
            if 'data' in first or 'traj' in first:
                for item in dataset:
                    steps = item.get('data', item.get('traj', []))
                    train_dataset.extend(steps)
            else:
                train_dataset = dataset
        else:
            train_dataset = dataset

    test_dataset = []
    train_dataset = datasets.Dataset.from_list(train_dataset)
    test_dataset = datasets.Dataset.from_list(test_dataset)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('prompt')

            answer = example.pop('response')
            data = {
                "data_source": "cold_start",
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "agent",
                "reward_model": {
                    "style": "instruction_following",
                    "ground_truth": question,
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer,
                    "question": question,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
