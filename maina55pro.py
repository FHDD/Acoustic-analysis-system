import sys
from gui import Ui_Form
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *

import pyaudio
import wave

import time

import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import pywt


import numpy as np

from scipy.io import wavfile, loadmat
from hmmlearn import hmm
from sklearn.externals import joblib
import os
# from Wavelet import *


from PyQt5 import QtMultimedia
from PyQt5.QtCore import QUrl



plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

class mygui(QtWidgets.QWidget,Ui_Form):
    def __init__(self):
        super(mygui,self).__init__()
        self.setupUi(self)

    def button_clickrecord(self):
        self.textEdit.setText('开始录音')

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 16000
        RECORD_SECONDS = 1
        WAVE_OUTPUT_FILENAME = "youngboy.wav"

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)


        frames = []


        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        self.textEdit.setText('')
        self.textEdit.setText('录音结束')

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()


    def button_clicklisten(self):
        sb = soundBase('youngboy.wav')
        data, fs = sb.audioread()

        sb_c = soundBase('younggirl.wav')
        # *******************
        nstr=self.textEdit_2.toPlainText()
        # print(type(nstr))
        # n=int(nstr)
        # print(type(n))
        # print('ok')
        n=eval(nstr)
        n+=1
        # *******************
        sb_c.audiowrite(data, fs * n)

        sb_c.audioplayer()


    def button_clickshow(self):
        CATEGORY = ['1', '2', '3']
        models = Model(CATEGORY=CATEGORY)
        models.load()

        # sb = soundBase(r'C:\Users\hufei\Documents\Corel VideoStudio Pro\21.0\a53recordandshowpro\youngboy20.wav')
        sb = soundBase('youngboy.wav')

        data, fs = sb.audioread()
        data = data[:, 0]
        res=models.testhalfself(data)

        self.textEdit_3.setText(res[0])


        # self.F = MyFigure(width=30, height=2, dpi=100)
        #
        # sb = soundBase('youngboy.wav')
        # data, fs = sb.audioread()
        # # ************************************
        # tempp = np.zeros(len(data))
        # tempp = data - np.mean(data)
        # data = tempp  # 损失了soudrate信息
        # # ************************************
        # # data -= np.mean(data)
        # # ************************************
        # tempp = np.zeros(len(data))
        # tempp = data / np.max(np.abs(data))
        # data = tempp  # 损失了soudrate信息
        # # ************************************
        # # print(data.shape[0])
        # # print(len(data))
        # tm = [i / fs for i in range(data.shape[0])]
        # self.F.axes1.plot(tm,data[:,0])
        # self.F.axes1.set_ylabel('原信号')
        #
        #
        # # wname = 'db7'
        # # jN = 6
        # # res_s = Wavelet_Soft(data, jN, wname)
        # # self.F.axes2.plot(tm,res_s[:,0])
        #
        #
        #
        #
        # N = len(data)
        # time = [i / fs for i in range(N)]
        # SNR = 5
        #
        # data=data[:,0]  # add
        #
        # r1 = awgn(data, SNR)
        # M, mu = 64, 0.001
        # itr = len(r1)
        # snr1 = SNR_Calc(data, r1 - data)
        #
        # [y, W, e] = LMS(r1, data, M, mu, itr)
        # output = e / np.max(np.abs(e))
        #
        # self.F.axes2.plot(tm, output)
        # self.F.axes2.set_ylabel('LMS滤波')
        #
        #
        #
        # self.gridlayout = QGridLayout(self.groupBox)  # 继承容器groupBox
        # self.gridlayout.addWidget(self.F,0,1)


    def button_choose(self):
        global imgName

        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", r"E:\研究生\HW\声学\sound\speech-demo-master\python\audio", "All Files(*)")
        file=QUrl.fromLocalFile(imgName)
        # file = QUrl.fromLocalFile('文本转声音.mp3')  # 音频文件路径
        content = QtMultimedia.QMediaContent(file)
        player = QtMultimedia.QMediaPlayer()
        player.setMedia(content)
        player.setVolume(50.0)
        player.play()
        time.sleep(6)  # 设置延时等待音频播放结束


    def button_recg(self):
        global imgName
        print('ok')
        CATEGORY = ['1', '2', '3']
        models = Model(CATEGORY=CATEGORY)
        models.load()

        # sb = soundBase(r'C:\Users\hufei\Documents\Corel VideoStudio Pro\21.0\a53recordandshowpro\youngboy20.wav')
        # sb = soundBase('youngboy.wav')
        sb = soundBase(imgName)


        data, fs = sb.audioread()
        data = data[:, 0]
        res=models.testhalfself(data)

        self.textEdit_4.setText(res[0])



class Model:
    def __init__(self, CATEGORY=None, n_comp=3, n_mix=3, cov_type='diag', n_iter=1000):
        super(Model, self).__init__()
        self.CATEGORY = CATEGORY
        self.category = len(CATEGORY)
        self.n_comp = n_comp
        self.n_mix = n_mix
        self.cov_type = cov_type
        self.n_iter = n_iter
        # 关键步骤，初始化models，返回特定参数的模型的列表
        self.models = []
        # self.models = self.load()

    def load(self, path="onetwosixmodels.pkl"):
        self.models = joblib.load(path)

    def testhalfself(self, thsdata):
        result = []
        mfcc = Nmfcc(thsdata, 8000, 24, 256, 80)

        result_one = []
        for m in range(self.category):
            model = self.models[m]
            re = model.score(mfcc)
            result_one.append(re)
        result.append(self.CATEGORY[np.argmax(np.array(result_one))])

        if result == ['3']:  # add
            result = ['6']

        print('识别得到结果：\n', result)
        return result




def Nmfcc(x, fs, p, frameSize, inc, nfft=512, n_dct=12):
    """
    计算mfcc系数
    :param x: 输入信号
    :param fs: 采样率
    :param p: Mel滤波器组的个数
    :param frameSize: 分帧的每帧长度
    :param inc: 帧移
    :return:
    """
    # 预处理-预加重
    xx = lfilter([1, -0.9375], [1], x)
    # 预处理-分幀
    xx = enframe(xx, frameSize, inc)
    # 预处理-加窗
    xx = np.multiply(xx, np.hanning(frameSize))
    # 计算FFT
    xx = np.fft.rfft(xx, nfft)
    # 计算能量谱
    xx = (np.abs(xx) ** 2) / nfft
    # 计算通过Mel滤波器的能量
    bank = melbankm(p, nfft, fs, 0, 0.5 * fs, 0)
    ss = np.matmul(xx, bank.T)
    # 计算DCT倒谱
    M = bank.shape[0]
    m = np.array([i for i in range(M)])
    mfcc = np.zeros((ss.shape[0], n_dct))
    for n in range(n_dct):
        mfcc[:, n] = np.sqrt(2 / M) * np.sum(np.multiply(np.log(ss), np.cos((2 * m - 1) * n * np.pi / 2 / M)), axis=1)
    return mfcc

def enframe(x, win, inc=None):
    nx = len(x)
    if isinstance(win, list) or isinstance(win, np.ndarray):
        nwin = len(win)
        nlen = nwin  # 帧长=窗长
    elif isinstance(win, int):
        nwin = 1
        nlen = win  # 设置为帧长
    if inc is None:
        inc = nlen
    nf = (nx - nlen + inc) // inc
    frameout = np.zeros((nf, nlen))
    indf = np.multiply(inc, np.array([i for i in range(nf)]))
    for i in range(nf):
        frameout[i, :] = x[indf[i]:indf[i] + nlen]
    if isinstance(win, list) or isinstance(win, np.ndarray):
        frameout = np.multiply(frameout, np.array(win))
    return frameout

def melbankm(p, NFFT, fs, fl, fh, w=None):
    """
    计算Mel滤波器组
    :param p: 滤波器个数
    :param n: 一帧FFT后的数据长度
    :param fs: 采样率
    :param fl: 最低频率
    :param fh: 最高频率
    :param w: 窗(没有加窗，无效参数)
    :return:
    """
    bl = 1125 * np.log(1 + fl / 700)  # 把 Hz 变成 Mel
    bh = 1125 * np.log(1 + fh / 700)
    B = bh - bl  # Mel带宽
    y = np.linspace(0, B, p + 2)  # 将梅尔刻度等间隔
    Fb = 700 * (np.exp(y / 1125) - 1)  # 把 Mel 变成Hz
    W2 = int(NFFT / 2 + 1)
    df = fs / NFFT
    freq = [int(i * df) for i in range(W2)]  # 采样频率值
    bank = np.zeros((p, W2))
    for k in range(1, p + 1):
        f0, f1, f2 = Fb[k], Fb[k - 1], Fb[k + 1]
        n1 = np.floor(f1 / df)
        n2 = np.floor(f2 / df)
        n0 = np.floor(f0 / df)
        for i in range(1, W2):
            if n1 <= i <= n0:
                bank[k - 1, i] = (i - n1) / (n0 - n1)
            elif n0 < i <= n2:
                bank[k - 1, i] = (n2 - i) / (n2 - n0)
            elif i > n2:
                break
        # plt.plot(freq, bank[k - 1, :], 'r')
    # plt.savefig('images/mel.png')
    return bank

def Wavelet_Soft(s, jN, wname):
    """
    小波软阈值滤波
    :param s:
    :param jN:
    :param wname:
    :return:
    """
    ca, cd = wavedec(s, jN, wname)
    for i in range(len(ca)):
        thr = np.median(cd[i] * np.sqrt(2 * np.log((i + 2) / (i + 1)))) / 0.6745
        di = np.array(cd[i])
        cd[i] = np.where(np.abs(di) > thr, np.sign(di) * (np.abs(di) - thr), 0)
    calast = np.array(ca[-1])
    thr = np.median(calast * np.sqrt(2 * np.log((jN + 1) / jN))) / 0.6745
    calast = np.where(np.abs(calast) > thr, np.sign(calast) * (np.abs(calast) - thr), 0)
    cd.append(calast)
    coef = cd[::-1]
    res = pywt.waverec(coef, wname)
    return res

def wavedec(s, jN, wname):
    ca, cd = [], []
    a = s
    for i in range(jN):
        a, d = pywt.dwt(a, wname)
        ca.append(a)
        cd.append(d)
    return ca, cd


def SNR_Calc(s, r):
    """
    计算信号的信噪比
    :param s: 信号
    :param r1: 噪声
    :return:
    """
    Ps = np.sum(np.power(s - np.mean(s), 2))
    Pr = np.sum(np.power(r - np.mean(r), 2))
    return 10 * np.log10(Ps / Pr)


def LMS(xn, dn, M, mu, itr):
    """
    使用LMS自适应滤波
    :param xn:输入的信号序列
    :param dn:所期望的响应序列
    :param M:滤波器的阶数
    :param mu:收敛因子(步长)
    :param itr:迭代次数
    :return:
    """
    en = np.zeros(itr)  # 误差序列,en(k)表示第k次迭代时预期输出与实际输入的误差
    W = np.zeros((M, itr))  # 每一行代表一个加权参量,每一列代表-次迭代,初始为0
    # 迭代计算
    for k in range(M, itr):
        x = xn[k:k - M:-1]
        y = np.matmul(W[:, k - 1], x)
        en[k] = dn[k] - y
        W[:, k] = W[:, k - 1] + 2 * mu * en[k] * x
    # 求最优输出序列
    yn = np.inf * np.ones(len(xn))
    for k in range(M, len(xn)):
        x = xn[k:k - M:-1]
        yn[k] = np.matmul(W[:, -1], x)
    return yn, W, en

def awgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return x + np.random.randn(len(x)) * np.sqrt(npower)



class soundBase:
    def __init__(self, path):
        self.path = path

    def audiorecorder(self, len=2, formater=pyaudio.paInt16, rate=16000, frames_per_buffer=1024, channels=2):
        """
        使用麦克风进行录音
        2020-2-25   Jie Y.  Init
        :param len: 录制时间长度(秒)
        :param formater: 格式
        :param rate: 采样率
        :param frames_per_buffer:
        :param channels: 通道数
        :return:
        """
        p = pyaudio.PyAudio()
        stream = p.open(format=formater, channels=channels, rate=rate, input=True, frames_per_buffer=frames_per_buffer)
        print("start recording......")
        frames = []
        for i in range(0, int(rate / frames_per_buffer * len)):
            data = stream.read(frames_per_buffer)
            frames.append(data)
        print("stop recording......")
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(self.path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(formater))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()

    def audioplayer(self, frames_per_buffer=1024):
        """
        播放语音文件
        2020-2-25   Jie Y.  Init
        :param frames_per_buffer:
        :return:
        """
        wf = wave.open(self.path, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        data = wf.readframes(frames_per_buffer)
        while data != b'':
            stream.write(data)
            data = wf.readframes(frames_per_buffer)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def audiowrite(self, data, fs, binary=True, channel=1, path=[]):
        """
        信息写入到.wav文件中
        :param data: 语音信息数据
        :param fs: 采样率(Hz)
        :param binary: 是否写成二进制文件(只有在写成二进制文件才能用audioplayer播放)
        :param channel: 通道数
        :param path: 文件路径，默认为self.path的路径
        :return:
        """
        if len(path) == 0:
            path = self.path
        if binary:
            wf = wave.open(path, 'wb')
            wf.setframerate(fs)
            wf.setnchannels(channel)
            wf.setsampwidth(2)
            wf.writeframes(b''.join(data))
        else:
            wavfile.write(path, fs, data)

    def audioread(self, return_nbits=False, formater='sample'):
        """
        读取语音文件
        2020-2-26   Jie Y.  Init
        这里的wavfile.read()函数修改了里面的代码，返回项return fs, data 改为了return fs, data, bit_depth
        如果这里报错，可以将wavfile.read()修改。
        :param formater: 获取数据的格式，为sample时，数据为float32的，[-1,1]，同matlab同名函数. 否则为文件本身的数据格式
                        指定formater为任意非sample字符串，则返回原始数据。
        :return: 语音数据data, 采样率fs，数据位数bits
        """
        # fs, data, bits = wavfile.read(self.path)
        # if formater == 'sample':
        #     data = data / (2 ** (bits - 1))
        # if return_nbits:
        #     return data, fs, bits
        # else:
        #     return data, fs

        fs, data = wavfile.read(self.path)
        return data, fs


    def soundplot(self, data=[], sr=16000, size=(14, 5)):
        """
        将语音数据/或读取语音数据并绘制出来
        2020-2-25   Jie Y.  Init
        :param data: 语音数据
        :param sr: 采样率
        :param size: 绘图窗口大小
        :return:
        """
        if len(data) == 0:
            # data, fs, _ = self.audioread()
            data, fs = self.audioread()
        plt.figure(figsize=size)
        x = [i / sr for i in range(len(data))]
        plt.plot(x, data)
        plt.xlim([0, len(data) / sr])
        plt.xlabel('s')
        plt.show()

    def sound_add(self, data1, data2):
        """
        将两个信号序列相加，若长短不一，在短的序列后端补零
        :param data1: 序列1
        :param data2: 序列2
        :return:
        """
        if len(data1) < len(data2):
            tmp = np.zeros([len(data2)])
            for i in range(len(data1)):
                tmp[i] += data1[i]
            return tmp + data2
        elif len(data1) > len(data2):
            tmp = np.zeros([len(data1)])
            for i in range(len(data2)):
                tmp[i] += data2[i]
            return tmp + data1
        else:
            return data1 + data2

    def SPL(self, data, fs, frameLen=100, isplot=True):
        """
        计算声压曲线
        2020-2-26   Jie Y.  Init
        :param data: 语音信号数据
        :param fs: 采样率
        :param frameLen: 计算声压的时间长度(ms单位)
        :param isplot: 是否绘图，默认是
        :return: 返回声压列表spls
        """

        def spl_cal(s, fs, frameLen):
            """
            根据数学公式计算单个声压值
            $y=\sqrt(\sum_{i=1}^Nx^2(i))$
            2020-2-26   Jie Y. Init
            :param s: 输入数据
            :param fs: 采样率
            :param frameLen: 计算声压的时间长度(ms单位)
            :return: 单个声压数值
            """
            l = len(s)
            M = frameLen * fs / 1000
            if not l == M:
                exit('输入信号长度与所定义帧长不等！')
            # 计算有效声压
            pp = 0
            for i in range(int(M)):
                pp += (s[i] * s[i])
            pa = np.sqrt(pp / M)
            p0 = 2e-5
            spl = 20 * np.log10(pa / p0)
            return spl

        length = len(data)
        M = fs * frameLen // 1000
        m = length % M
        if not m < M // 2:
            # 最后一帧长度不小于M的一半
            data = np.hstack((data, np.zeros(M - m)))
        else:
            # 最后一帧长度小于M的一半
            data = data[:M * (length // M)]
        spls = np.zeros(len(data) // M)
        for i in range(length // M - 1):
            s = data[i * M:(i + 1) * M]
            spls[i] = spl_cal(s, fs, frameLen)

        if isplot:
            plt.subplot(211)
            plt.plot(data)
            plt.subplot(212)
            plt.step([i for i in range(len(spls))], spls)
            plt.show()
        return spls

    def iso226(self, phon, isplot=True):
        """
        绘制等响度曲线，输入响度phon
        2020-2-26   Jie Y.  Init
        :param phon: 响度值0~90
        :param isplot: 是否绘图，默认是
        :return:
        """
        ## 参数来源: 语音信号处理试验教程，梁瑞宇P36-P37
        f = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, \
             1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500]
        af = [0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330, 0.315, \
              0.301, 0.288, 0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244, 0.243, 0.243, \
              0.243, 0.242, 0.242, 0.245, 0.254, 0.271, 0.301]

        Lu = [-31.6, - 27.2, - 23.0, - 19.1, - 15.9, - 13.0, - 10.3, - 8.1, - 6.2, - 4.5, - 3.1, \
              - 2.0, - 1.1, - 0.4, 0.0, 0.3, 0.5, 0.0, - 2.7, - 4.1, - 1.0, 1.7, \
              2.5, 1.2, - 2.1, - 7.1, - 11.2, - 10.7, - 3.1]

        Tf = [78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, \
              11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7, - 1.3, - 4.2, \
              - 6.0, - 5.4, - 1.5, 6.0, 12.6, 13.9, 12.3]
        if phon < 0 or phon > 90:
            print('Phon value out of range!')
            spl = 0
            freq = 0
        else:
            Ln = phon
            # 从响度级计算声压级
            Af = 4.47E-3 * (10 ** (0.025 * Ln) - 1.15) + np.power(0.4 * np.power(10, np.add(Tf, Lu) / 10 - 9), af)
            Lp = np.multiply(np.divide(10, af), np.log10(Af)) - Lu + 94
            spl = Lp
            freq = f
            if isplot:
                plt.semilogx(freq, spl, ':k')
                plt.axis([20, 20000, -10, 130])
                plt.title('Phon={}'.format(phon))
                plt.grid()
                plt.show()
        return spl, freq

    def vowel_generate(self, len, pitch=100, sr=16000, f=[730, 1090, 2440]):
        """
        生成一个元音片段
        2020-2-26   Jie Y.  Init
        :param len: 长度，点数
        :param pitch:
        :param sr: 采样率
        :param f: 前3个共振峰，默认为元音a的
        :return: 生成的序列
        """
        f1, f2, f3 = f[0], f[1], f[2]
        y = np.zeros(len)
        points = [i for i in range(0, len, sr // pitch)]
        indices = np.array(list(map(int, np.floor(points))))
        y[indices] = (indices + 1) - points
        y[indices + 1] = points - indices

        a = np.exp(-250 * 2 * np.pi / sr)
        y = lfilter([1], [1, 0, -a * a], y)
        if f1 > 0:
            cft = f1 / sr
            bw = 50
            q = f1 / bw
            rho = np.exp(-np.pi * cft / q)
            theta = 2 * np.pi * cft * np.sqrt(1 - 1 / (4 * q * q))
            a2 = -2 * rho * np.cos(theta)
            a3 = rho * rho
            y = lfilter([1 + a2 + a3], [1, a2, a3], y)

        if f2 > 0:
            cft = f2 / sr
            bw = 50
            q = f2 / bw
            rho = np.exp(-np.pi * cft / q)
            theta = 2 * np.pi * cft * np.sqrt(1 - 1 / (4 * q * q))
            a2 = -2 * rho * np.cos(theta)
            a3 = rho * rho
            y = lfilter([1 + a2 + a3], [1, a2, a3], y)
        if f3 > 0:
            cft = f3 / sr
            bw = 50
            q = f3 / bw
            rho = np.exp(-np.pi * cft / q)
            theta = 2 * np.pi * cft * np.sqrt(1 - 1 / (4 * q * q))
            a2 = -2 * rho * np.cos(theta)
            a3 = rho * rho
            y = lfilter([1 + a2 + a3], [1, a2, a3], y)
        plt.plot(y)
        plt.show()
        return y


class MyFigure(FigureCanvas):
    def __init__(self,width=5, height=4, dpi=100):
        #第一步：创建一个创建Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        #第二步：在父类中激活Figure窗口
        super(MyFigure,self).__init__(self.fig) #此句必不可少，否则不能显示图形
        #第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        # self.axes1 = self.fig.add_subplot(311)
        # self.axes2 = self.fig.add_subplot(312)
        # self.axes3 = self.fig.add_subplot(313)
        self.axes1 = self.fig.add_subplot(211)
        self.axes2 = self.fig.add_subplot(212)






if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    a = mygui()
    a.show()
    sys.exit(app.exec_())