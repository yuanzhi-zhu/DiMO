import math
import os
import random
import torch
# import json

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info


class ImageDataset(IterableDataset):
    def __init__(
        self,
        path,
        resolution,
        random_crop=False,
        random_flip=0.0,
        prompt_only=False,
        tokenizer=None,
        shuffle=True,
        ratings="all",
    ):
        super().__init__()

        self.name = 'aesthetics'
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.prompt_only = prompt_only
        assert self.prompt_only, 'prompt_only must set to True for Lion_aesthetics dataset.'
        self.prompt_list = []
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        
        
        # filenames = ['aesthetics_6_plus.txt', 'aesthetics_625_plus.txt', 'aesthetics_65_plus.txt']
        if ratings == 'all':
            filenames = ['aesthetics_6_plus.txt', 'aesthetics_625_plus.txt', 'aesthetics_65_plus.txt']
        elif ratings == '6':
            filenames = ['aesthetics_6_plus.txt'],
        elif ratings == '625':
            filenames = ['aesthetics_625_plus.txt', 'aesthetics_65_plus.txt']
        elif ratings == '65':
            filenames = ['aesthetics_65_plus.txt'],

        # Check if any of the preferred files exists in the specified path
        for filename in filenames:
            full_path = os.path.join(path, filename)
            if os.path.exists(full_path):
                break
        with open(full_path, 'rt') as f:
            print(full_path)
            for row in f:
                ##
                row = row.strip('\n')
                ##
                self.prompt_list.append(row)


    def __len__(self):
        return len(self.prompt_list)

    # def __getitem__(self, idx):
    #     return torch.zeros(1, 4, 4), self.prompt_list[idx]  # Return a dummy image when prompt_only is set.
    
    def __iter__(self):
        worker_info = get_worker_info()
        
        if worker_info is None:  # single-process data loading, return the full iterator
            data_list = self.prompt_list
        else:
            len_data = len(self.prompt_list) - len(self.prompt_list) % worker_info.num_workers
            data_list = self.prompt_list[:len_data][worker_info.id :: worker_info.num_workers]
            # print(worker_info.num_workers, worker_info.id, len(data_list)/len(self.prompt_list))
            
        if self.shuffle:
            random.shuffle(data_list) 
            
        while True:
            for idx in range(len(data_list)):
                # try:
                # shard_name = data_list[idx][0]
                # shard_name = data_list[idx]["shard"]
                data = {}
                
                # img_file = data_list[idx][1]
                # img_file = data_list[idx]["img"]
                # img = Image.open(os.path.join(self.img_root, shard_name, img_file)).convert("RGB")
                
                data['pixel_values'] = torch.zeros(1, 4, 4)
                
                text = data_list[idx]
                if self.tokenizer is not None:
                    if isinstance(self.tokenizer, list):
                        assert len(self.tokenizer)>=1
                        data['input_ids'] = self.tokenizer[0](text)[0]
                        # data['input_ids_2'] = self.tokenizer[1](text)[0]
                        for i in range(1, len(self.tokenizer)):
                            key = 'input_ids_' + str(i+1)
                            data[key] = self.tokenizer[i](text)[0]
                    else:
                        data['input_ids'] = self.tokenizer(text)[0]
                else:
                    data['input_ids'] = text
                
                yield data
                
                # except Exception as e:
                #     raise(e)

    def collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        
        if self.tokenizer is not None:
            if isinstance(self.tokenizer, list):
                assert len(self.tokenizer)>=1
                input_ids = torch.stack([example["input_ids"] for example in examples])
                # input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])
                # return {"pixel_values": pixel_values, "input_ids": input_ids, "input_ids_2": input_ids_2,}
                output = {"pixel_values": pixel_values, "input_ids": input_ids,}
                for i in range(1, len(self.tokenizer)):
                    key = 'input_ids_' + str(i+1)
                    output[key] = torch.stack([example[key] for example in examples])
                return output

            else:
                input_ids = torch.stack([example["input_ids"] for example in examples])
                return {"pixel_values": pixel_values, "input_ids": input_ids,}
        else:
            input_ids = [example["input_ids"] for example in examples]
            # return {"pixel_values": pixel_values, "input_ids": input_ids,}
            return {"input_ids": input_ids,}
    


def make_train_dataset(
        train_data_path, 
        size = 512,
        tokenizer=None, 
        tokenizer_max_length = None,
        cfg_drop_ratio=0,
        rank=0, 
        world_size=1,
        shuffle=True,
        ratings='all',
    ):

    train_dataset = ImageDataset(
        path=train_data_path,
        resolution=size,
        shuffle=shuffle,
        prompt_only=True,
        ratings=ratings,
    )
    
    return train_dataset
