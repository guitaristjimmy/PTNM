from __future__ import print_function
from __future__ import division
from multiprocessing import Process, Manager
import os
import librosa
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import cv2
import ffmpeg   # read mp3
import openpyxl

class Arithmet(object):
    def __init__(self):
        self.cumul_p = []

    def normalize_data(self, data):
        sum = 0
        for i in range(0, len(data)):
            sum += data[i]
        for i in range(0, len(data)):
            data[i] = data[i]/sum

        return data

    def cal_cumul_p(self, p):
        # 누적 확률 계산 / p = [0.1, 0.1, 0.1, 0.1, 0.6]
        sum = 0
        for i in range(0, len(p)):
            sum += p[i]
            sum = round(sum, 15)
            self.cumul_p.append(sum)

    def arith(self, interval):
        len_interval = interval[1] - interval[0]
        # print(interval, len_interval)
        interval = [interval[0], len_interval*self.cumul_p[0]]
        len_interval = interval[1] - interval[0]
        # print(interval, len_interval)
        for i in range(1, len(self.cumul_p)):
            interval = [len_interval*self.cumul_p[i-1]+interval[0], len_interval*self.cumul_p[i]+interval[0]]
            len_interval = round(interval[1] - interval[0], 15)
            # print(interval, len_interval)

        arith_code = round((interval[0]+interval[1])/2, 15)
        return arith_code


class PTNM_Feature_Extract(Arithmet):
    def __init__(self):
        self.name_path = []
        self.audio_data = []
        self.samplerate = []
        self.file_list = []
        self.file_list_name = []
        ##histograms (bins, ranges and channels H and S )기본 range 설정값 넣어주기
        self.h_bins = 50
        self.s_bins = 60
        self.histSize = [self.h_bins, self.s_bins]
        # hue varies from 0 to 179, saturation from 0 to 255
        self.h_ranges = [0, 180]
        self.s_ranges = [0, 256]
        self.ranges = self.h_ranges + self.s_ranges
        # Use the 0-th and 1-st channels
        self.channels = [0, 1]
        self.mfcc_vec = []
        self.base_vec = np.eye(12)
        super().__init__()


    def name_load(self, folder_path):  # y 값에 load 시키기 전에 경로+음원 이름 설정하기
        self.folder_list = os.listdir(folder_path)
        print('Folder : ', self.folder_list[0], '~', self.folder_list[-1])  # 폴더 이름 확인
        total_path = []
        for k in range(len(self.folder_list)):
            total_path.append(folder_path + '/' + self.folder_list[k])
            self.file_list.append(os.listdir(total_path[k]))
            # print(total_path[k])
            # print(self.file_list[k])
            for i in range(len(self.file_list[k])):
                self.name_path.append(total_path[k] + '/' + self.file_list[k][i])
                self.file_list_name.append(self.file_list[k][i])
                self.audio_data.append(0)
                self.samplerate.append(0)
        #print(self.name_path)

    def audio_read(self, start, end):
        for i in range(start, end):
            print('audio_read : ', self.file_list_name[i])
            self.audio_data[i], self.samplerate[i] = librosa.load(self.name_path[i], offset=30, duration=90)

            print('audio_read done')

        # print(self.samplerate, self.audio_data)

    def chromaExtract(self, i):
        self.chroma = librosa.feature.chroma_stft(self.audio_data[i], sr=self.samplerate[i])
        chroma_img = librosa.display.specshow(self.chroma)
        fig_name = './Feature_Img/Chroma/' + self.file_list_name[i][:-4] + '_Chroma_result.png'
        plt.savefig(fig_name, bbox_inches='tight', dpi=500, frameon='false')

    def tonnetzExtract(self, i):
        self.tonnet = librosa.feature.tonnetz(self.audio_data[i], sr=self.samplerate[i], chroma=self.chroma)
        tonnetz_img = librosa.display.specshow(self.tonnet)
        fig_name = './Feature_Img/Tonnetz/' + self.file_list_name[i][:-4] + '_Tonnetz_result.png'
        plt.savefig(fig_name, bbox_inches='tight', dpi=500, frameon='false')

    def sCentroidExtract(self, i):
        self.sCentroid = librosa.feature.spectral_centroid(self.audio_data[i], sr=self.samplerate[i])
        Centroid_img = librosa.display.specshow(self.sCentroid)
        fig_name = './Feature_Img/sCentroid/' + self.file_list_name[i][:-4] + '_sCentro_result.png'
        plt.savefig(fig_name, bbox_inches='tight', dpi=500, frameon='false')

    def sBandwidthExtract(self, i):
        self.sBW = librosa.feature.spectral_bandwidth(self.audio_data[i], sr=self.samplerate[i])
        BW_img = librosa.display.specshow(self.sBW)
        fig_name = './Feature_Img/sBW/' + self.file_list_name[i][:-4] + '_sBW_result.png'
        plt.savefig(fig_name, bbox_inches='tight', dpi=500, frameon='false')

    def sContraExtract(self, i):
        self.sC = librosa.feature.spectral_contrast(self.audio_data[i], sr=self.samplerate[i])
        Contra_img = librosa.display.specshow(self.sC)
        fig_name = './Feature_Img/sContra/' + self.file_list_name[i][:-4] + '_sContra_result.png'
        plt.savefig(fig_name, bbox_inches='tight', dpi=500, frameon='false')

    def mel_specto_extract(self, i):
        mel_specto = librosa.feature.melspectrogram(self.audio_data[i], sr=self.samplerate[i], n_mels=128)
        mel_specto_img = librosa.display.specshow(librosa.power_to_db(mel_specto, ref=np.max))
        fig_name = './Feature_Img/Mel_specto/' + self.file_list_name[i][:-4] + '_Mel_specto_result.png'
        plt.savefig(fig_name, bbox_inches='tight', dpi=500, frameon='false')

    def mfcc_extract(self, i):
        mfcc = librosa.feature.mfcc(self.audio_data[i], sr=self.samplerate[i], n_mfcc=12)
        temp = []

        for i in range(0, 12):
            temp.append(mfcc[i].mean())

        self.mfcc_vec.append(temp)

    def mfcc2csv(self, name, mfcc_vec, interval, iter_id):
        arith_data = []
        for i in range(0, len(mfcc_vec)):
            temp = super().normalize_data(mfcc_vec[i])
            super().cal_cumul_p(temp)
            arith_data.append(super().arith(interval))
            self.cumul_p = []
        result = pd.DataFrame(data=arith_data, index=name)
        path = './DB/mfcc/'+iter_id+'_mfcc.csv'
        result.to_csv(path_or_buf=path, mode='a', header=False, encoding='utf-8-sig')

    def cal_img_mean(self, img_list):
        img_mean_list = []
        for i in range(0, len(img_list)):
            img_mean_list.append(img_list[i].mean())
        return img_mean_list

    def cal_img_rank(self, img_list):
        rank = pd.Series(img_list)
        return rank.rank()

    def chroma_crop(self, img):
        crop = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        crop[0] = img[54:203, 77:2551]
        crop[1] = img[205:357, 77:2251]
        crop[2] = img[359:511, 77:2251]
        crop[3] = img[513:665, 77:2251]
        crop[4] = img[667:819, 77:2251]
        crop[5] = img[821:973, 77:2251]
        crop[6] = img[975:1127, 77:2251]
        crop[7] = img[1129:1281, 77:2251]
        crop[8] = img[1283:1435, 77:2251]
        crop[9] = img[1437:1589, 77:2251]
        crop[10] = img[1591:1743, 77:2251]
        crop[11] = img[1745:1894, 77:2251]

        return crop
        # 자른 파일을 저장하고 싶다면 아래 코드를 사용해라
        # cv2.imwrite('chroma_result-1.jpg', crop[0])
        # cv2.imwrite('chroma_result-2.jpg', crop[1])
        # cv2.imwrite('chroma_result-3.jpg', crop[2])
        # cv2.imwrite('chroma_result-4.jpg', crop[3])
        # cv2.imwrite('chroma_result-5.jpg', crop[4])
        # cv2.imwrite('chroma_result-6.jpg', crop[5])
        # cv2.imwrite('chroma_result-7.jpg', crop[6])
        # cv2.imwrite('chroma_result-8.jpg', crop[7])
        # cv2.imwrite('chroma_result-9.jpg', crop[8])
        # cv2.imwrite('chroma_result-10.jpg', crop[9])
        # cv2.imwrite('chroma_result-11.jpg', crop[10])
        # cv2.imwrite('chroma_result-12.jpg', crop[11])

    def tonnetz_crop(self, img):
        t_crop = [0, 0, 0, 0, 0, 0]
        t_crop[0] = img[54:357, 77:2551]
        t_crop[1] = img[359:665, 77:2251]
        t_crop[2] = img[667:973, 77:2251]
        t_crop[3] = img[975:1281, 77:2251]
        t_crop[4] = img[1283:1589, 77:2251]
        t_crop[5] = img[1591:1894, 77:2251]

        return t_crop
        # 자른 파일을 저장하고 싶다면 아래 코드를 사용해라
        # cv2.imwrite('tonnetz_result-1.jpg', t_crop[0])
        # cv2.imwrite('tonnetz_result-2.jpg', t_crop[1])
        # cv2.imwrite('tonnetz_result-3.jpg', t_crop[2])
        # cv2.imwrite('tonnetz_result-4.jpg', t_crop[3])
        # cv2.imwrite('tonnetz_result-5.jpg', t_crop[4])
        # cv2.imwrite('tonnetz_result-6.jpg', t_crop[5])

    def ChromaHist(self, base_path, chroma_crop_img):

        # 기준base데이터 크로마그램사진으로 읽어오기-->보라색 가져오거라
        base_img = cv2.imread(base_path)

        # 기준데이터 자르기
        # base_c_crop_img=[]
        base_c_crop_img = self.chroma_crop(base_img)

        # 기준 데이터 가중치 구하기
        cal_cr_mean = self.cal_img_mean(base_c_crop_img)
        print(cal_cr_mean)
        base_cr_rank = self.cal_img_rank(cal_cr_mean)
        print(base_cr_rank)

        # Convert them to HSV format:
        hsv_base = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        hsv_test = []

        # hist list만들기
        hist_base = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        hist_test = []

        baseVStest = []

        for k in range(len(self.file_list_name)):
            hsv_test.append([])
            hist_test.append([])
            baseVStest.append([])
            for t in range(0, 12):
                hsv_base[t] = cv2.cvtColor(base_c_crop_img[t], cv2.COLOR_BGR2HSV)
                hsv_test[k].append(cv2.cvtColor(chroma_crop_img[k][t], cv2.COLOR_BGR2HSV))

                hist_base[t] = cv2.calcHist([hsv_base[t]], self.channels, None, self.histSize, self.ranges,
                                            accumulate=False)
                cv2.normalize(hist_base[t], hist_base[t], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                hist_test[k].append(cv2.calcHist([hsv_test[k][t]], self.channels, None, self.histSize, self.ranges,
                                                 accumulate=False))
                cv2.normalize(hist_test[k][t], hist_test[k][t], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                # print(hist_test)

                compare_hist = cv2.compareHist(hist_base[t], hist_test[k][t], 4)
                compare_weight = (compare_hist) * (base_cr_rank[t])
                baseVStest[k].append(compare_weight)

        # print('(compare_Method=Chi-square) : ', baseVStest[0], '\n', '\t', '\t', baseVStest[1])
        self.vs2excel(path='./red_flavor_alt_chai_Chroma_Hist_Compare.xlsx',
                      baseVStest=baseVStest)
            ##크로마엑셀 저장할 위치 및 이름 설정해주기

    def tonnetzHist(self, base_path, tonnetz_crop_img):

        # 기준base데이터 토넷츠사진으로 읽어오기-->그레이색으로 가져오거라
        img_base = cv2.imread(base_path)

        # 기준데이터 자르기
        # base_t_crop_img=[]
        base_t_crop_img = (self.tonnetz_crop(img_base))

        # Convert them to HSV format:
        hsv_base = [0, 0, 0, 0, 0, 0]
        hsv_test = []

        # hist list만들기
        hist_base = [0, 0, 0, 0, 0, 0]
        hist_test = []
        baseVStest = []

        for k in range(len(self.file_list_name)):
            hsv_test.append([])
            hist_test.append([])
            baseVStest.append([])
            for t in range(0, 6):
                hsv_base[t] = cv2.cvtColor(base_t_crop_img[t], cv2.COLOR_BGR2HSV)
                hsv_test[k].append(cv2.cvtColor(tonnetz_crop_img[k][t], cv2.COLOR_BGR2HSV))

                hist_base[t] = cv2.calcHist([hsv_base[t]], self.channels, None, self.histSize, self.ranges,
                                            accumulate=False)
                cv2.normalize(hist_base[t], hist_base[t], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                hist_test[k].append(
                    cv2.calcHist([hsv_test[k][t]], self.channels, None, self.histSize, self.ranges, accumulate=False))
                cv2.normalize(hist_test[k][t], hist_test[k][t], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                # print(hist_test)

                baseVStest[k].append(cv2.compareHist(hist_base[t], hist_test[k][t], 4))
            # print('(compare_Method=Chi-square) : ', baseVStest[0], '\n', '\t', '\t', baseVStest[1])
        ##토넷츠 엑셀 저장할 위치 및 이름 설정해주기
        self.vs2excel(path='./red_flavor_alt_chai_Tonnetz_Hist_Compare.xlsx',
                      baseVStest=baseVStest)

    def vs2excel(self, path, baseVStest):
        compare = pd.DataFrame.from_records(baseVStest, index=[self.file_list_name])
        compare.to_excel(path)

    def readKorean(self, i):
        src_base = open(self.name_path[i], "rb")
        bytes = bytearray(src_base.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        return bgrImage

    def extract_ChromaTonnetz(self, start, end):
        for i in range(start, end):
            self.chromaExtract(i)
            print('chroma done : ' + str(i))
            self.tonnetzExtract(i)
            print('tonnetz done : ' + str(i))

    def extract_spactralFeature(self, start, end):
        for i in range(start, end):
            self.sCentroidExtract(i)
            print('Spactral_Centroid done : ', i)
            self.sBandwidthExtract(i)
            print('Spectral_Bandwidth done : ', i)

    def extract_sContra_mel(self, start, end):
        for i in range(start, end):
            self.sContraExtract(i)
            print('Spectral_Contrast done : ', i)
            self.mel_specto_extract(i)
            print('Mel_specto_extract done : ', i)

    def compare_cos(self, target_vec):
        cos_value = []
        for i in range(0, len(self.base_vec)):
            cos_value.append(dot(self.base_vec[i], target_vec) / (norm(self.base_vec[i]) * norm(target_vec))+1)
            # cos유사도의 원래 범위는 -1 ~ 1 까지이지만 0~2까지로 변경
        return cos_value


class Mfcc_Ext2csv(PTNM_Feature_Extract):
    def __init__(self):
        super().__init__()

    def name_load(self, folder_path):  # y 값에 load 시키기 전에 경로+음원 이름 설정하기
        self.folder_list = os.listdir(folder_path)
        # print('Folder : ', self.folder_list[0], '~', self.folder_list[-1])  # 폴더 이름 확인

        total_path = []
        for k in range(len(self.folder_list)):
            total_path.append(folder_path + '/' + self.folder_list[k])
            self.file_list.append(os.listdir(total_path[k]))
            # print(total_path[k])
            # print(self.file_list[k])
            self.name_path.append([])
            self.file_list_name.append([])
            self.audio_data.append([])
            self.samplerate.append([])
            for i in range(len(self.file_list[k])):
                self.name_path[k].append(total_path[k] + '/' + self.file_list[k][i])
                self.file_list_name[k].append(self.file_list[k][i][:-4])
                self.audio_data[k].append(0)
                self.samplerate[k].append(0)
        #print(self.name_path)

    def audio_read(self, start, end, iter_id):

        for i in range(start, end):
            print('audio_read : ', self.file_list_name[iter_id][i])
            self.audio_data[iter_id][i], self.samplerate[iter_id][i] = librosa.load(self.name_path[iter_id][i], offset=30, duration=90)
            print('audio_read done')
        # print(self.samplerate, self.audio_data)

    def mfcc_extract(self, i, j):
        mfcc = librosa.feature.mfcc(self.audio_data[i][j], sr=self.samplerate[i][j], n_mfcc=12)
        mfcc_vec = []
        for i in range(0, 12):
            mfcc_vec.append(mfcc[i].mean()+500)

        return mfcc_vec

    def action_function(self, start, end, iter_id):
        self.audio_read(start=start, end=end, iter_id=iter_id)
        # print(start, end, 'audio_data[iter_id] :: ', self.audio_data)

        mfcc_vec_list = []
        for i in range(start, end):
            # print(start, end, 'file_list_name :: ', self.file_list_name[iter_id][i])
            mfcc_vec_list.append(self.mfcc_extract(i=iter_id, j=i))
        csv_iter_id = self.folder_list[iter_id]
        # print(mfcc_vec_list)
        super().mfcc2csv(name=self.file_list_name[iter_id][start:end], mfcc_vec=mfcc_vec_list, interval=[0, 100], iter_id=csv_iter_id)

        for i in range(start, end):
            self.audio_data[iter_id][i] = 0
            self.samplerate[iter_id][i] = 0

    def mfcc_ext_process(self, end_len, id):

        for i in range(0, end_len):
            start = 10*i
            end = start + 10
            if end >= len(self.name_path[id]):
                end = len(self.name_path[id])

            self.action_function(start, end, id)


if __name__ == '__main__':

    folder_path = 'C:/Users/K/Desktop/I_SW/Python_Note/PTNM_Feature_Extract/audio/Total_Audio'

    # mfcc_cos유사도----------------------------------------------------------------------------------------------------

    mfcc_ext = Mfcc_Ext2csv()
    mfcc_ext.name_load(folder_path=folder_path)

    p_mng = Manager()
    for i in range(0, int(len(mfcc_ext.name_path)/4)+1):

        s_flag_02 = 0
        s_flag_03 = 0
        s_flag_04 = 0

        p_end_01 = int(len(mfcc_ext.name_path[4 * i]) / 10) + 1
        pros_01 = Process(target=mfcc_ext.mfcc_ext_process, args=(p_end_01, 4*i))
        pros_01.start()
        print('pros_01 start')

        if 4*i+1 < len(mfcc_ext.name_path):
            p_end_02 = int(len(mfcc_ext.name_path[4 * i+1]) / 10) + 1
            pros_02 = Process(target=mfcc_ext.mfcc_ext_process, args=(p_end_02, 4*i+1))
            pros_02.start()
            s_flag_02 = 1
            print('pros_02 start')

        if 4*i+2 < len(mfcc_ext.name_path):
            p_end_03 = int(len(mfcc_ext.name_path[4 * i+2]) / 10) + 1
            pros_03 = Process(target=mfcc_ext.mfcc_ext_process, args=(p_end_03, 4*i+2))
            pros_03.start()
            print('pros_03 start')
            s_flag_03 = 1

        if 4*i+3 < len(mfcc_ext.name_path):
            p_end_04 = int(len(mfcc_ext.name_path[4 * i+3]) / 10) + 1
            pros_04 = Process(target=mfcc_ext.mfcc_ext_process, args=(p_end_04, 4*i+3))
            pros_04.start()
            print('pros_04 start')
            s_flag_04 = 1

        pros_01.join()
        print('pros_01 join')
        if s_flag_02 == 1:
            pros_02.join()
            print('pros_02 join')
        if s_flag_03 == 1:
            pros_03.join()
            print('pros_03 join')
        if s_flag_04 == 1:
            pros_04.join()
            print('pros_04 join')

    #
    # ext.audio_read(0, len(ext.file_list_name))
    # vec_list = []
    # for i in range(0, len(ext.file_list_name)):
    #     ext.mfcc_extract(i)
    #     vec_list.append(ext.compare_cos(target_vec=ext.mfcc_vec[i]))
    # print(vec_list)
    # ext.mfcc2csv(name=ext.file_list_name, mfcc_vec=vec_list, interval=[0, 100], iter_id='test')
    # ------------------------------------------------------------------------------------------------------------------

    # Extract Feature Image---------------------------------------------------------------------------------------------
    #
    # ext = PTNM_Feature_Extract()
    # ext.name_load(folder_path)
    #
    # for i in range(0, int(len(ext.file_list_name)/10)+1):
    #
    #     start = i*10
    #     end = start + 10
    #     if end >= len(ext.file_list_name):
    #         end = len(ext.file_list_name)
    #
    #     print('Audio Read Start')
    #     print('start = ', start, 'end = ', end)
    #
    #     ext.audio_read(start, end)
    #
    #     print('Feature Extract Start')
    #     print('start = ', start, 'end = ', end)
    #
    #     pros01 = Process(target=ext.extract_ChromaTonnetz, args=(start, end))
    #     pros02 = Process(target=ext.extract_spactralFeature, args=(start, end))
    #     pros03 = Process(target=ext.extract_sContra_mel, args=(start, end))
    #
    #     pros01.start()
    #     print('process01 start')
    #     pros02.start()
    #     print('process02 start')
    #     pros03.start()
    #     print('process03 start')
    #
    #     pros01.join()
    #     pros02.join()
    #     pros03.join()
    #
    #     for j in range(0, len(ext.file_list_name)):
    #         ext.audio_data[j] = 0
    #         ext.samplerate[j] = 0
    # ------------------------------------------------------------------------------------------------------------------
