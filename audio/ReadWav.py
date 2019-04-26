import wave
import numpy as np

from util.config import logger


class ReadWav(object):
    def __init__(self):
        pass

    def readWavFile(self, wav_path):
        """
        :param wav_path: 需要读取的文件路径
        :return: （声道，量化位数，采样率，帧数，数据)
        """
        read_file = wave.open(wav_path, "rb")
        params = read_file.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        total_time = int(nframes / framerate * 1000)  # wav文件时长(ms)
        # logger.info("{0} 总时长 {1} ms".format(wav_path, total_time))
        data = read_file.readframes(nframes)
        wave_data = np.fromstring(data, dtype=np.short)
        if nchannels == 2:
            wave_data = wave_data[0:int(nframes * nchannels):2]
        if nframes >= 1000000:
            nframes = 1000000
            wave_data = wave_data[:1000000]
        wave_data = wave_data / 32768  # 归一化
        return nchannels, sampwidth, framerate, nframes, wave_data
