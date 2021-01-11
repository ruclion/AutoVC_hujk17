import os
import torch
import numpy as np
import pickle as pkl
from torch.utils import data

 

speaker_emb_dict = pkl.load(open('/ceph/home/hujk17/AutoVC_hujk17/full_106_spmel_nosli/speaker_embs_dict_full_106_nosli.pkl', 'rb'))



def text2list(file):
    f = open(file, 'r').readlines()
    file_list = [i.strip() for i in f]
    return file_list


def get_single_data_pair(fpath, speaker_name):
    # print('mel-path:', fpath)
    mel = np.load(fpath)
    speaker_emb = speaker_emb_dict[speaker_name]
    return mel, speaker_emb



class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, root_dir, meta_path, max_len):
        """Initialize and preprocess the Utterances dataset."""
        self.root_dir = root_dir
        self.max_len = max_len
        self.file_list = text2list(file=meta_path)
        
        
    def __getitem__(self, index):
        now = self.file_list[index].split('|')
        mel, speaker_emb = get_single_data_pair(os.path.join(self.root_dir, now[0]), now[1])


        if mel.shape[0] < self.max_len:
            len_pad = self.max_len - mel.shape[0]
            mel_fix = np.pad(mel, ((0,len_pad),(0,0)), 'constant')
        elif mel.shape[0] > self.max_len:
            left = np.random.randint(mel.shape[0]-self.max_len + 1)
            assert left + self.max_len <= mel.shape[0]
            mel_fix = mel[left:left+self.max_len, :]
        else:
            mel_fix = mel
        
        return mel_fix, speaker_emb
    

    def __len__(self):
        return len(self.file_list)
    
    
    

def get_loader(root_dir, meta_path, batch_size=16, max_len=128, shuffle=True, drop_last = False, num_workers=0):
    """Build and return a data loader."""
    
    dataset = Utterances(root_dir, meta_path, max_len)
    
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  drop_last=drop_last)
    return data_loader






