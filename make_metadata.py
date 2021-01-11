import os
import torch
import pickle
import numpy as np
from model_bl import D_VECTOR
from collections import OrderedDict


# in
speaker_encoder_pretrained_model = 'pretrained-3000000-BL.ckpt'
rootDir = './full_106_spmel_nosli'
use_speakers_path = '/ceph/dataset/VCTK-Corpus/vctk_metadata.tsv'

# out
# 这些txt也会放在rootDir中
train_meta_path = 'train_meta_full_106_nosli.txt'
val_meta_path = 'val_meta_full_106_nosli.txt'
unseen_meta_path = 'unseen_meta_full_106_nosli.txt'
# 这些embs也会放在rootDir中
speaker_embs_pkl = 'speaker_embs_dict_full_106_nosli.pkl'
speaker_seen_unseen_path = 'speaker_seen_unseen.txt'


num_uttrs = 10
len_crop = 128
np.random.seed(231)



def main():
    C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
    c_checkpoint = torch.load(speaker_encoder_pretrained_model)
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)
    

    use_speakers_f = open(use_speakers_path, 'r')
    use_speakers_a = [x.strip().split()[-1] for x in use_speakers_f.readlines()]


    speakers_embs = dict()
    speakers_status = dict()
    train_mel_speaker_list = []
    val_mel_speaker_list = []
    unseen_mel_speaker_list = []

    dirName, subdirList, _ = next(os.walk(rootDir))
    for speaker in sorted(subdirList):
        if speaker not in use_speakers_a:
            continue
        print('Processing speaker: %s' % speaker)
        _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
        
        # make speaker embedding
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
                # print('while ---:', tmp.shape[0], len_crop)
            # print('final ---:', tmp.shape[0], len_crop)
            left = np.random.randint(0, tmp.shape[0] - len_crop + 1)
            assert left + len_crop <= tmp.shape[0]
            melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
            emb = C(melsp)
            embs.append(emb.detach().squeeze().cpu().numpy())  

        speaker_avg_emb = np.mean(embs, axis=0)
        speakers_embs[speaker] = speaker_avg_emb


        if np.random.uniform() < 0.9:
            speakers_status[speaker] = 'seen'
            for fileName in sorted(fileList):
                if np.random.uniform() < 0.9:
                    train_mel_speaker_list.append(os.path.join(speaker,fileName) + '|' + speaker)
                else:
                    val_mel_speaker_list.append(os.path.join(speaker,fileName) + '|' + speaker)
        else:
            speakers_status[speaker] = 'unseen'
            for fileName in sorted(fileList):
                unseen_mel_speaker_list.append(os.path.join(speaker,fileName) + '|' + speaker)

        # break


    with open(os.path.join(rootDir, speaker_embs_pkl), 'wb') as f:
        pickle.dump(speakers_embs, f) 
    with open(os.path.join(rootDir, speaker_seen_unseen_path), 'w') as f:
        for key in speakers_status:
            f.write(key + '|' + speakers_status[key] + '\n')
    

    with open(os.path.join(rootDir, train_meta_path), 'w') as f:
        for x in train_mel_speaker_list:
            f.write(x + '\n')
    with open(os.path.join(rootDir, val_meta_path), 'w') as f:
        for x in val_mel_speaker_list:
            f.write(x + '\n')
    with open(os.path.join(rootDir, unseen_meta_path), 'w') as f:
        for x in unseen_mel_speaker_list:
            f.write(x + '\n')


if __name__ == "__main__":
    main()