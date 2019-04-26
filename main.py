import csv
import cv2
import hashlib
import json
import math
import os
import random
import http.client
import numpy as np
import pyaudio
import pandas as pd

from audio.ReadWav import ReadWav
from util.fanyi import en_to_zh


def piecewise(data, winfunc=lambda x: np.ones((x,))):
    """
    处理音频数据，将其分成part_num部分
    :param winfunc:
    :param data:
    :return:
    """
    frame_time = 10  # 多少ms一帧(ms)
    frame_step = frame_time / 2  # 帧的步长
    nchannels, sampwidth, framerate, nframes, wave_data = data
    signal_length = nframes  # 信号总长度
    frame_length = int(round(framerate / 1000 * frame_time))  # 以帧帧时间长度
    frame_step = int(round(framerate / 1000 * frame_step))  # 相邻帧之间的步长
    if signal_length <= frame_length:  # 若信号长度小于一个帧的长度，则帧数定义为1
        frames_num = 1
    else:  # 否则，计算帧的总长度
        frames_num = 1 + int(math.ceil((1.0 * signal_length - frame_length) / frame_step))
    pad_length = int((frames_num - 1) * frame_step + frame_length)  # 所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((wave_data, zeros))  # 填补后的信号记为pad_signal
    x = np.arange(0, frame_length)
    y = np.arange(0, frames_num * frame_step, frame_step)
    a = np.tile(x, (frames_num, 1))
    b = np.tile(y, (frame_length, 1))
    bt = b.T
    indices = a + bt  # 相当于对所有帧的时间点进行抽取，得到frames_num*frame_length长度的矩阵
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    t = winfunc(frame_length)
    win = np.tile(t, (frames_num, 1))  # window窗函数，这里默认取1
    return frames * win  # 返回帧信号矩阵

def audioToSpectrogram(frames, n=2000):
    """
    音频信号转为语谱图
    :param n:
    :param frames:
    :return:
    """
    complex_spectrum = np.fft.rfft(frames, n=n * 2)
    amp_spectrum = np.absolute(complex_spectrum)
    phase = np.angle(complex_spectrum)
    spec = np.log1p(amp_spectrum)
    return amp_spectrum, spec, phase


def play(wave_data, sampwidth, nchannels, framerate):
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


def princen_bradley(x):
    winfunc = np.sin((np.pi / 2) * np.power(np.sin(np.pi * np.arange(0, x) / x), 2))
    return winfunc


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
        # play(wave_data, sampwidth, nchannels, framerate)
        matrix = piecewise((nchannels, sampwidth, framerate, nframes, wave_data),
                                        winfunc=princen_bradley)
        spectrogram = audioToSpectrogram(matrix)
        cv2.imshow('', cv2.resize(spectrogram[0], (1920, 1080)))
        cv2.waitKey(0)
        print()


if __name__ == '__main__':
    main()
