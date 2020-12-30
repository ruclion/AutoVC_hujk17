import os
import numpy as np
from scipy.io import wavfile
import librosa



# audio file directory
rootDir = '/ceph/dataset/VCTK-Corpus/wav16'
# spectrogram directory
targetDir = '/ceph/home/hujk17/VCTK-Corpus/wav16_nosli'



def get_best_trim(y):
    index = [librosa.effects.trim(y, i)[1] for i in range(20, 60, 5)]
    starts, ends = list(zip(*index))

    diff_start = np.diff(starts)
    best_start = min(zip(diff_start, starts))[1]

    diff_end = np.diff(ends)
    best_end = max(zip(diff_end, ends))[1]
    return best_start, best_end


def main():
    dirName, subdirList, _ = next(os.walk(rootDir))
    print('Found directory: %s' % dirName)

    for subdir in sorted(subdirList):
        print(subdir)
        if not os.path.exists(os.path.join(targetDir, subdir)):
            os.makedirs(os.path.join(targetDir, subdir))
        _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
        for fileName in sorted(fileList):
            try:
                # Read audio file
                wav_path = os.path.join(dirName,subdir,fileName)
                # Calcu
                save_wav_nosli_path = os.path.join(targetDir, subdir, fileName)
                y, sr = librosa.load(wav_path)
                start, end = get_best_trim(y)
                yy = y[start: end]
                # Save
                yy *= 32767 / np.max(np.abs(yy))
                wavfile.write(save_wav_nosli_path, sr, yy.astype(np.int16))
                # break
            except Exception as e:
                print('Reason:', str(e))  
        # break


if __name__ == "__main__":
    main()