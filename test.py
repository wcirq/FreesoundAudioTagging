import csv
import hashlib
import json
import os
import random
import http.client
import numpy as np
import pyaudio
import pandas as pd

from audio.ReadWav import ReadWav
from util.fanyi import en_to_zh


def main():
    root = "C:/MySystem/ FreesoundAudioTagging/freesound-audio-tagging-2019/"

    test_file_path = "C:/MySystem/ FreesoundAudioTagging/freesound-audio-tagging-2019/test"
    wav_noisy_path = "C:/MySystem/ FreesoundAudioTagging/freesound-audio-tagging-2019/train_noisy"
    wav_curated_path = "C:/MySystem/ FreesoundAudioTagging/freesound-audio-tagging-2019/train_curated"
    wav_noisy_label = "C:/MySystem/ FreesoundAudioTagging/freesound-audio-tagging-2019/train_noisy.csv"
    wav_curated_label = "C:/MySystem/ FreesoundAudioTagging/freesound-audio-tagging-2019/train_curated.csv"
    sample_submission = "C:/MySystem/ FreesoundAudioTagging/freesound-audio-tagging-2019/sample_submission.csv"

    wav_file_list = []
    for path in [wav_noisy_path, wav_curated_path]:
        for file_name in os.listdir(path):
            file_name_path = os.path.join(path.split("/")[-1], file_name)
            wav_file_list.append(file_name_path)

    label_noisy = pd.read_csv(wav_noisy_label)
    label_curated = pd.read_csv(wav_curated_label)

    label_data = pd.concat([label_noisy, label_curated], ignore_index=True)
    a = label_data.groupby(['labels']).size().reset_index()
    name_list = set()
    for names in a.values[:, 0]:
        names = names.split(",")
        for name in names:
            name_list.add(name)
    print(name_list)
    label_data = label_data.values
    labels = {}
    for k, v in zip(label_data[:, 0], label_data[:, 1]):
        labels[k] = v
    readwav = ReadWav()
    for file_name in wav_file_list:
        type_name = labels[os.path.basename(file_name)]
        names = type_name.split(",")
        print(file_name)
        for name in names:
            type_name1 = name.replace("_", " ")
            print(name, en_to_zh(name), "\n", type_name1, en_to_zh(type_name1))
        file_path = os.path.join(root, file_name)
        nchannels, sampwidth, framerate, nframes, wave_data = readwav.readWavFile(file_path)
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(sampwidth),
                        channels=nchannels,
                        rate=framerate,
                        output=True)
        wave_data = np.asarray(wave_data * 32768, np.short)
        wave_data = np.maximum(np.minimum(wave_data, 32767), -32768)
        data = wave_data[:1600].tobytes()
        i = 1
        while data != b'':
            if i > 50:
                break
            stream.write(data)
            data = wave_data[int(i * 1600):int(i * 1600 + 1600)].tobytes()
            i += 1
        print()


if __name__ == '__main__':
    main()
