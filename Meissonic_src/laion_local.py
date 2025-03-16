import os
import json
import random
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageStat
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from diffusers.training_utils import resolve_interpolation_mode
from torchvision import transforms


### >>>>>>>> >>>>>>>> text related >>>>>>>> >>>>>>>> ###

class TokenizerWrapper():
    def __init__(self, tokenizer, is_train, proportion_empty_prompts, use_generic_prompts=False, model_max_length=None):
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.proportion_empty_prompts = proportion_empty_prompts
        self.use_generic_prompts = use_generic_prompts
        self.model_max_length = model_max_length

    def __call__(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        captions = []
        for caption in prompts:
            if random.random() < self.proportion_empty_prompts:
                captions.append("")
            else:
                if self.use_generic_prompts:
                    captions.append("best quality, high quality")
                elif isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(random.choice(caption) if self.is_train else caption[0])
                else:
                    raise ValueError(
                        f"Caption column should contain either strings or lists of strings."
                    )
        inputs = self.tokenizer(
            captions, max_length=self.model_max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        return inputs.input_ids



### >>>>>>>> >>>>>>>> image related >>>>>>>> >>>>>>>> ###

MONOCHROMATIC_MAX_VARIANCE = 0.3

def is_monochromatic_image(pil_img):
    v = ImageStat.Stat(pil_img.convert('RGB')).var
    return sum(v)<MONOCHROMATIC_MAX_VARIANCE

def isnumeric(text):
    return (''.join(filter(str.isalnum, text))).isnumeric()


class TextPromptDataset_Json(IterableDataset):
    '''
      The dataset for (text embedding, noise, generated latent) triplets.
    '''
    def __init__(self, 
                data_root, 
                tokenizer = None,
                # transform = None,
                rank = 0,
                world_size = 1,
                resolution = 512,
                shuffle = True,
    ):
        self.tokenizer = tokenizer
        # self.transform = transform
        self.resolution = resolution
        self.shuffle = shuffle
        
        print("#### Loading filename list...")
        # json_file = "/cpfs/data/user/yuazhu/codes/perflow-dev/filtered_data.json"
        json_file = "/cpfs/data/user/yuazhu/codes/perflow_related/filtered_data_401489.json"
        # json_file = "/cpfs/data/user/yuazhu/codes/perflow_related/test_data.json"
        # json_file = "/cpfs/data/user/yuazhu/codes/perflow_related/filtered_data_4011168.json"
        
        with open(json_file, "r") as f:
            data_list = json.load(f)
        
        # duplicate several shards to make sure each process has the same number of shards
        assert len(data_list) >= world_size, f"len(data_list)={len(data_list)}, world_size={world_size}"
        duplicate = world_size - len(data_list)%world_size if len(data_list)%world_size>0 else 0
        data_list = data_list + data_list[:duplicate]
        self.data_list = data_list[rank::world_size]

        self.img_root = os.path.join(data_root, 'images')
        print("#### All filename loaded...")
        print(f'#### Number of data: {len(self.data_list)} on rank {rank}...')

    def __len__(self):
        return len(self.data_list)
    
    
    def __iter__(self):
        worker_info = get_worker_info()
        
        if worker_info is None:  # single-process data loading, return the full iterator
            data_list = self.data_list
        else:
            len_data = len(self.data_list) - len(self.data_list) % worker_info.num_workers
            data_list = self.data_list[:len_data][worker_info.id :: worker_info.num_workers]
            # print(worker_info.num_workers, worker_info.id, len(data_list)/len(self.data_list))
            
        if self.shuffle:
            random.shuffle(data_list) 
        
        use_fix_crop_and_size = False
        interpolation_type = 'bilinear'
        interpolation_mode = resolve_interpolation_mode(interpolation_type)
        
        while True:    
            for idx in range(len(data_list)):
                # try:
                # shard_name = data_list[idx][0]
                # shard_name = data_list[idx]["shard"]
                data = {}
                
                # img_file = data_list[idx][1]
                # img_file = data_list[idx]["img"]
                # img = Image.open(os.path.join(self.img_root, shard_name, img_file)).convert("RGB")
                img_file = data_list[idx]["path"]
                img = Image.open(os.path.join(self.img_root, img_file)).convert("RGB")
                img_WIDTH, img_HEIGHT = img.size
                data['orig_size'] = (img_WIDTH, img_HEIGHT)
                
                if is_monochromatic_image(img):
                    continue
                
                # resize image
                img = TF.resize(img, self.resolution, interpolation=interpolation_mode)
                img_WIDTH_, img_HEIGHT_ = img.size
                # get crop coordinates and crop image
                # c_top, c_left, _, _ = transforms.CenterCrop.get_params(img, output_size=(self.resolution, self.resolution))
                c_top = int((img_HEIGHT_ - self.resolution) / 2)
                c_left = int((img_WIDTH_ - self.resolution) / 2)
                img = TF.crop(img, c_top, c_left, self.resolution, self.resolution)

                # Apply sharpening
                img = TF.adjust_sharpness(img, sharpness_factor=3)
                img = TF.to_tensor(img)
                img = TF.normalize(img, [0.5], [0.5])

                data["crop_coords"] = (c_top, c_left) if not use_fix_crop_and_size else (0, 0)
    
                # if self.transform is not None:
                #     img = self.transform(data)
                    
                data['pixel_values'] = img
                
                text = data_list[idx]["caption"]
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
        orig_sizes = [example["orig_size"] for example in examples]
        crop_coords = [example["crop_coords"] for example in examples]
        return pixel_values, input_ids, orig_sizes, crop_coords
    

class TextPromptDataset(IterableDataset):
    '''
      The dataset for (text embedding, noise, generated latent) triplets.
    '''
    def __init__(self, 
                data_root, 
                tokenizer = None,
                transform = None,
                rank = 0,
                world_size = 1,
                shuffle = True,
    ):
        self.tokenizer = tokenizer
        self.transform = transform
        self.shuffle = shuffle
        
        print("#### Loading filename list...")
        if data_root.endswith("__"):
            data_root = data_root[: -1*len("__")]
            json_root = os.path.join(data_root, 'list')
        elif data_root.endswith("_recaption"):
            data_root = data_root[: -1*len("_recaption")]
            json_root = os.path.join(data_root, 'list_recaption')
        elif '_recap_' in data_root:
            # data_root = data_root[: -1*len("_recaption")]
            json_root = os.path.join(data_root, 'annotations')
        # elif data_root.endswith("_?"):
        else:
            raise NotImplementedError
        # json_list = [os.path.join(root, file) for root, _, files in os.walk(json_root) for file in files if file.endswith('.json')]
        json_list = []
        for json_split in os.listdir(json_root):
            for number_json in os.listdir(os.path.join(json_root, json_split)):
                for filename in os.listdir(os.path.join(json_root, json_split, number_json)):
                    if filename.endswith('.json'):
                        json_list.append(os.path.join(json_split, number_json, filename))
                break
            break
        print(f'WARNING: Only loading the first json file: {json_list[0]}')

        # json_list = [os.path.join(json_split, number_json, filename)
        #             for json_split in os.listdir(json_root)
        #             for number_json in os.listdir(os.path.join(json_root, json_split))
        #             for filename in os.listdir(os.path.join(json_root, json_split, number_json))
        #             if filename.endswith('.json')]
        # json_list = [p for p in os.listdir(json_root) if p.startswith("shard") and p.endswith('.json')]
        
        # duplicate several shards to make sure each process has the same number of shards
        assert len(json_list) > world_size
        duplicate = world_size - len(json_list)%world_size if len(json_list)%world_size>0 else 0
        json_list = json_list + json_list[:duplicate]
        json_list = json_list[rank::world_size]
        
        self.data_list = []
        for json_file in tqdm(json_list):
            # shard_name = os.path.basename(json_file).split('.')[0]
            # load json file, each line is a dictionary
            with open(os.path.join(json_root, json_file), "r") as f:
                items = [json.loads(line) for line in f]
            # with open(os.path.join(json_root, json_file)) as f:
            #     key_text_pairs = json.load(f)
            for item in items[0]:
                # item['shard'] = shard_name
                self.data_list.append(item)

        self.img_root = os.path.join(data_root, 'images')
        print("#### All filename loaded...")

        
    def __len__(self):
        return len(self.data_list)
    
    
    def __iter__(self):
        worker_info = get_worker_info()
        
        if worker_info is None:  # single-process data loading, return the full iterator
            data_list = self.data_list
        else:
            len_data = len(self.data_list) - len(self.data_list) % worker_info.num_workers
            data_list = self.data_list[:len_data][worker_info.id :: worker_info.num_workers]
            # print(worker_info.num_workers, worker_info.id, len(data_list)/len(self.data_list))
            
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
                img_file = data_list[idx]["path"]
                img = Image.open(os.path.join(self.img_root, img_file)).convert("RGB")
                
                if is_monochromatic_image(img):
                    continue
                
                if self.transform is not None:
                    img = self.transform(img)
                    
                data['pixel_values'] = img
                
                text = data_list[idx]["caption"]
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
            return {"pixel_values": pixel_values, "input_ids": input_ids,}
    
    
def make_train_dataset(
        train_data_path, 
        size = 512,
        tokenizer=None, 
        tokenizer_max_length = None,
        cfg_drop_ratio=0,
        rank=0, 
        world_size=1,
        shuffle=True,
    ):

    _image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(size),
            T.CenterCrop(size),
            T.RandomAdjustSharpness(sharpness_factor=3, p=1), #!!!: sharpening
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    if tokenizer is not None:
        if isinstance(tokenizer, list):
            assert len(tokenizer)>=1
            
            _tokenizer = []
            for tknz in tokenizer:
                if tokenizer_max_length is None:
                    _tokenizer_max_length = tknz.model_max_length
                else:
                    _tokenizer_max_length = tokenizer_max_length
                _tokenizer.append(
                    TokenizerWrapper(tknz, is_train=True, 
                                     proportion_empty_prompts=cfg_drop_ratio, 
                                     use_generic_prompts=False, 
                                     model_max_length=_tokenizer_max_length
                    ))
            tokenizer = _tokenizer
            
        else:
            tokenizer = TokenizerWrapper(
                tokenizer, 
                is_train=True, 
                proportion_empty_prompts=cfg_drop_ratio,
                use_generic_prompts=False,
            )

        
    train_dataset = TextPromptDataset_Json(
        data_root=train_data_path,
        # transform=_image_transform,
        resolution=size,
        rank=rank,
        world_size=world_size,
        tokenizer=tokenizer,
        shuffle=shuffle,
    )
    # train_dataset = TextPromptDataset(
    #     data_root=train_data_path,
    #     transform=_image_transform,
    #     rank=rank,
    #     world_size=world_size,
    #     tokenizer=tokenizer,
    #     shuffle=shuffle,
    # )
    return train_dataset
    
    
    
    
    
    
    
    
    

### >>>>>>>> >>>>>>>> Test >>>>>>>> >>>>>>>> ###
if __name__ == "__main__":
    from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer
    # tokenizer = CLIPTokenizer.from_pretrained(
    #     "/cpfs/data/user/yanghuan/huggingface/stabilityai/SD-XL-base-1.0",
    #     subfolder="tokenizer"
    # )
    # model_id = "/cpfs/data/user/yanghuan/huggingface/stabilityai/SD-XL-base-1.0"
    # tokenizer_one = AutoTokenizer.from_pretrained(
    #     model_id, subfolder="tokenizer", use_fast=False
    # )

    # tokenizer_two = AutoTokenizer.from_pretrained(
    #     model_id, subfolder="tokenizer_2", use_fast=False
    # )

    # tokenizers = [tokenizer_one, tokenizer_two]


    train_dataset = make_train_dataset(
        train_data_path = "/cpfs/data/user/chenbei/bohe/data/LAION_recap_0729/",
        size = 512,
        cfg_drop_ratio = 0,
        tokenizer=None, 
        rank=0, world_size=10,shuffle=False,
        )

    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, num_workers=0, 
        collate_fn=train_dataset.collect_fn if hasattr(train_dataset, 'collect_fn') else None,
    )

    batch = next(iter(loader))
    import pdb; pdb.set_trace()


    for batch in loader:
        pixel_values = batch["pixel_values"]
        # prompt_ids = batch['input_ids']
        text = batch['input_ids']
        from einops import rearrange
        pixel_values = rearrange(pixel_values, 'b c h w -> b h w c')
        
        for i in range(pixel_values.shape[0]):
            Image.fromarray(((pixel_values[i] + 1 )/2 * 255 ).numpy().astype(np.uint8)).save('tmp.png')
            # input_id = prompt_ids[i]
            # text = tokenizer_one.decode(input_id[0]).split('<|startoftext|>')[-1].split('<|endoftext|>')[0]
            print(text)
            import pdb; pdb.set_trace()
        pass