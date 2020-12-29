"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch
import copy

C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
c_checkpoint = torch.load('pretrained-3000000-BL.ckpt')
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)
num_uttrs = 10
len_crop = 128

# Directory containing mel-spectrograms
rootDir = './full_106_spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


use_speakers_path = '/ceph/dataset/VCTK-Corpus/vctk_metadata.tsv'
use_speakers_f = open(use_speakers_path, 'r')
use_speakers_a = [x.strip().split()[-1] for x in use_speakers_f.readlines()]



train_seen_speakers = []
val_seen_speakers = []
unseen_speakers = []
np.random.seed(231)
for speaker in sorted(subdirList):
    if speaker not in use_speakers_a:
        continue
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    
    # make speaker embedding
    assert len(fileList) >= num_uttrs
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    for i in range(num_uttrs):
        tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
        candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
        # choose another utterance if the current one is too short
        while tmp.shape[0] < len_crop:
            idx_alt = np.random.choice(candidates)
            tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
            candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
        left = np.random.randint(0, tmp.shape[0]-len_crop)
        melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
        emb = C(melsp)
        embs.append(emb.detach().squeeze().cpu().numpy())     
    utterances.append(np.mean(embs, axis=0))


    if np.random.uniform() < 0.9:
        train_utterances = copy.deepcopy(utterances)
        val_utterances = copy.deepcopy(utterances)
        # create file list
        for fileName in sorted(fileList):
            if np.random.uniform() < 0.9:
                train_utterances.append(os.path.join(speaker,fileName))
            else:
                val_utterances.append(os.path.join(speaker,fileName))
        train_seen_speakers.append(train_utterances)
        val_seen_speakers.append(val_utterances)
    else:
        # create file list
        for fileName in sorted(fileList):
            utterances.append(os.path.join(speaker,fileName))
        unseen_speakers.append(utterances)

    
with open(os.path.join(rootDir, 'train_seen_speaker.pkl'), 'wb') as handle:
    pickle.dump(train_seen_speakers, handle)
with open(os.path.join(rootDir, 'val_seen_speaker.pkl'), 'wb') as handle:
    pickle.dump(val_seen_speakers, handle)
with open(os.path.join(rootDir, 'unseen_speaker.pkl'), 'wb') as handle:
    pickle.dump(unseen_speakers, handle)

