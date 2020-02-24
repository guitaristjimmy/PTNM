import os
import librosa
import itertools
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import skew
import ffmpeg
import librosa.display
import openpyxl
import scipy.stats
from multiprocessing import Process, Queue
from multiprocessing import Pool, Manager
from numpy import dot
from numpy.linalg import norm
from collections import Counter


class Arithmet:

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
        sum = 0
        for i in range(0, len(p)):
            sum += p[i]
            sum = round(sum, 3)
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


class pkl2melody(Arithmet):

    def __init__(self):
        # 필요한 상수
        super().__init__()
        self.HopSize = 0.001 * 2.9  # 2.9 ms
        self.scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']  # 음계
        self.octave = [1, 2, 3, 4, 5, 6, 7, 8]  # 옥타브

        self.freq = np.arange(len(self.scale) * len(self.octave), dtype=float).reshape((12, 8))
        for K in range(len(self.scale)):
            for N in range(len(self.octave)):
                self.freq[K][N] = 2 ** N * 55 * 2 ** ((K - 9) / 12)  # 표준 주파수

    def open_pklmusic(self, pickle_name):  # data 꺼내기
        data = pd.read_pickle(pickle_name)
        data['bar_length'] = (60 / data['tempo']) * 4  # 1 마디 당 걸리는 시간
        count_music = len(data)
        print('pkl open success')
        return data, data['title']

    def division_data(self, data, i):  # 노래 한 곡씩 뽑기
        one_melody = pd.DataFrame(data['melody'][i], columns=['frequency'])
        plus_num = int(data['bar_length'][i] / self.HopSize)  # 합쳐야할 행의 개수
        return one_melody, plus_num

    def cleaned_melody(self, one_melody, plus_num):  # 음원 멜로디에서 앞 뒤 nan 값 제거

        start_index = one_melody.first_valid_index()
        final_index = one_melody.last_valid_index()

        cleaned_melody = one_melody[start_index:final_index]
        cleaned_melody['frame_num'] = range(len(cleaned_melody))
        cleaned_melody['bar_num'] = cleaned_melody.index // plus_num  # 한 음악 당 마디의 개수
        cleaned_melody = cleaned_melody.set_index(['bar_num', 'frame_num'])

        return cleaned_melody

    def division_melody(self, Cleanmelody, i):
        # 마디마다 분류
        bar_melody = Cleanmelody.loc[(i,), :]
        one_bar = list(bar_melody['frequency'].dropna())
        return one_bar

    def extract_chords(self, one_bar_melody):  # 한 마디씩 chords 추출
        temp = []
        midi = []
        last = []

        for i in range(len(one_bar_melody)):
            frequency = one_bar_melody[i]
            min = abs(self.freq[0][0] - frequency)
            for K in range(len(self.scale)):
                for N in range(len(self.octave)):
                    diff = abs(self.freq[K][N] - frequency)
                    # 표준 주파수 - 들어온 주파수 값
                    if min >= diff:  # 차이가 가장 작은 값을 찾기 위함
                        min = diff
                        a = self.scale[K]  # 음계
                        b = str(self.octave[N])  # 옥타브
                        c = (a, b)
            temp.append(c)

        for num in range(len(temp)):
            if num == 0:
                continue
            if temp[num] == temp[num - 1]:
                midi.append(temp[num])

        for num in range(len(midi)):
            if num == 0:
                last.append(midi[num])
                continue
            if midi[num] != midi[num - 1]:
                last.append(midi[num])

        return last

    def to_pickle(self, music_name, entire_music):
        df = pd.DataFrame({"title": music_name, "melody": entire_music})
        filename = os.path.basename(file_name)
        df.to_pickle(filename + "_melody.pkl", protocol=4)


class Code_Feature_Extract(pkl2melody):

    def __init__(self):
        # super = scale == >> ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        super(Code_Feature_Extract, self).__init__()
        self.base_vec = np.eye(12)
        self.file_list = []
        self.file_path = []
        # base_vec = np.eye(12) >> 12*12 대각행렬로 각 행이 단위 벡터가 된다.

    def file_load(self, folder_path):  # y 값에 load 시키기 전에 경로+음원 이름 설정하기
        self.file_list = os.listdir(folder_path)
        print('file : ', self.file_list[0], '~', self.file_list[-1])  # 폴더 이름 확인
        total_path = []
        for k in range(len(self.file_list)):
            total_path.append(folder_path + '/' + self.file_list[k])
            self.file_path.append(total_path[k])
            # print(total_path[k])
            # print(self.file_list[k])
        # print(self.name_path)

    def find_key(self, melody):
        #송희가 추출한 형태의 멜로디 모양을 기준으로 만듦.
        #[[마디1],[마디2]] 형태로 마디에 [('C',4),('C#',4),('F',5)]와 같이 데이터가 들어가 있다고 생각함.
        total_notes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(0, len(melody)):
            for j in range(0, len(melody[i])):
                for k in range(0, len(self.scale)):
                    total_notes[k] += melody[i][j][0].count(self.scale[k])

        note_count = pd.Series(data=total_notes, index=self.scale)
        # print(note_count)
        return note_count.idxmax()

    def melody2harmony(self, key, bar):
        if key != 'C':
            key_value = self.scale.index(key)
            harmony_index = self.scale[key_value:12]+self.scale[0:key_value]
            # print(harmony_index)
        else:
            harmony_index = self.scale

        harmony_num = []
        for i in range(0, len(bar)):
            harmony_num.append(harmony_index.index(bar[i][0]))
            # melody 문자열을 화성 조표에 따른 숫자로 저장한다.

        return harmony_num

    def harmony_num_count(self, harmony_list):
        harmony_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(0, len(harmony_list)):
            for j in range(0, len(harmony_list[i])):
                index_num = harmony_list[i][j]
                harmony_vec[index_num] += 1
        return harmony_vec

    def harmony2numchord(self, harmony_bar):
        pass

    def octave_num_melo(self, melody):
        octave_num_melo = []
        for i in range(0, len(melody)):
            for j in range(0, len(melody[i])):
                octave_num_melo.append(self.scale.index(melody[i][j][0])+(melody[i][j][1]*12))
        return octave_num_melo

    def cal_grad_melody(self, melody):
        melo_data = self.octave_num_melo(melody)

        grad_data = []
        for i in range(0, len(melo_data)-1):
            grad_data.append(melo_data[i+1] - melo_data[i])

        return grad_data

    def compare_cos(self, target_vec):
        cos_value = []
        for i in range(0, len(self.base_vec)):
            cos_value.append(dot(self.base_vec[i], target_vec) / (norm(self.base_vec[i]) * norm(target_vec)))

        return super().normalize_data(cos_value)

    def cos_v2arithmet(self, cos_v, interval):
        Arithmet.cal_cumul_p(self, p=cos_v)
        result = Arithmet.arith(self, interval)
        self.cumul_p = []

        return result

    def data2excel(self, path, data, index_list):
        compare = pd.DataFrame(data, index=[index_list])
        compare.to_excel(path)
        print('excel writing done')


class Action(pkl2melody):
    def __init__(self):
        super().__init__()

    def act(self, start, end, information, entire_music):

        for music_num in range(start, end):
            one_melody, plus_num = super().division_data(information, music_num)
            Cleanmelody = super().cleaned_melody(one_melody, plus_num)
            Barcount_max = max(Cleanmelody.index)[0]
            Barcount_min = min(Cleanmelody.index)[0]

            entire_melody = []
            for bar_count in range(Barcount_min, Barcount_max + 1):
                one_bar_melody = super().division_melody(Cleanmelody, bar_count)
                # print(one_bar_melody)
                # print(music_num)
                last_melody = super().extract_chords(one_bar_melody)
                entire_melody.append(last_melody)

            # print(entire_melody)
            entire_music[music_num] = entire_melody
            print('freq trans 2 note done : ', music_num)


if __name__ == '__main__':

    path = './pickle'

    cf_ext = Code_Feature_Extract()
    cf_ext.file_load(folder_path=path)
    for file in range(0, len(cf_ext.file_list)):
        information, music_name = cf_ext.open_pklmusic(cf_ext.file_path[file])
        mng = Manager()
        # print('length :: ', len(music_name))
        entire_music = mng.list(np.zeros(len(music_name)))
        act = Action()
        pros_01 = Process(target=act.act, args=(0, int(len(music_name)/4), information, entire_music))
        pros_02 = Process(target=act.act, args=(int(len(music_name)/4), int(len(music_name)/2), information, entire_music))
        pros_03 = Process(target=act.act, args=(int(len(music_name)/2), 3*int(len(music_name)/4), information, entire_music))
        pros_04 = Process(target=act.act, args=(3*int(len(music_name)/4), int(len(music_name)), information, entire_music))

        pros_01.start()
        print('pros_01 start :: ')
        pros_02.start()
        print('pros_02 start :: ')
        pros_03.start()
        print('pros_03 start :: ')
        pros_04.start()
        print('pros_04 start :: ')

        pros_01.join()
        print('pros_01 join :: ')
        pros_02.join()
        print('pros_02 join :: ')
        pros_03.join()
        print('pros_03 join :: ')
        pros_04.join()
        print('pros_04 join :: ')

        # print('entire_music == ', entire_music)
        # for music_num in range(len(music_name)):
        #     one_melody, plus_num = cf_ext.division_data(information, music_num)
        #     Cleanmelody = cf_ext.cleaned_melody(one_melody, plus_num)
        #     Barcount_max = max(Cleanmelody.index)[0]
        #     Barcount_min = min(Cleanmelody.index)[0]
        #
        #     entire_melody = []
        #     for bar_count in range(Barcount_min, Barcount_max + 1):
        #         one_bar_melody = cf_ext.division_melody(Cleanmelody)
        #         # print(one_bar_melody)
        #         # print(music_num)
        #         last_melody = cf_ext.extract_chords(one_bar_melody)
        #         entire_melody.append(last_melody)
        #
        #     # print(entire_melody)
        #     entire_music.append(entire_melody)
        #     print('freq trans 2 note done : ', music_num)

        # test_melody_data01 = [[('G#', 4), ('G', 4), ('G#', 4)],
        #                     [('G#', 4), ('F', 4), ('F#', 4), ('F', 4), ('F#', 4), ('G', 4), ('F#', 4), ('F', 4), ('E', 4),
        #                      ('D#', 4), ('G#', 4), ('A', 4), ('G#', 4), ('G', 4), ('G#', 4)],
        #                     [('B', 4), ('C', 5), ('C#', 5), ('C', 5), ('B', 4), ('A#', 4), ('A', 4), ('G#', 4), ('G', 4),
        #                      ('F#', 4), ('F', 4), ('F#', 4), ('F', 4), ('F#', 4), ('G',4), ('F#', 4)],
        #                     [('F', 4), ('E', 4), ('D#', 4), ('G#', 4), ('C', 4), ('B', 3), ('C', 4), ('B', 3), ('C', 4),
        #                      ('C#', 4), ('C', 4), ('C#', 4), ('D', 4), ('C#', 4), ('D', 4), ('C#', 4)]]
        # test_melody_data02 = [[('C', 4), ('E', 4), ('G', 4), ('C', 4), ('E', 4), ('G', 4)],
        #                       [('A', 4), ('G', 4), ('F', 4), ('E', 4), ('D', 4), ('C', 4)],
        #                       [('C', 0), ('C', 1), ('C', 2), ('C', 3)]]

        result = []
        for i in range(0, len(entire_music)):
            key = str(cf_ext.find_key(entire_music[i]))
            # print('이 곡의 key ==>> ', key)

            harmony_list = []
            for j in range(0, len(entire_music[i])):
                h_num = cf_ext.melody2harmony(key, entire_music[i][j])
                harmony_list.append(h_num)
                # print('화성으로 변환 ==>> ', h_num)

            # print('화성으로 변환한 전체 곡 ==>> ', harmony_list)

            h_vec = np.array(cf_ext.harmony_num_count(harmony_list=harmony_list))
            # print('전체 화성 개수 ==>> ', h_vec)

            cos_v = cf_ext.compare_cos(target_vec=h_vec)
            # print('cos유사도 분석 ==>> ', cos_v)
            result.append([cf_ext.cos_v2arithmet(cos_v=cos_v, interval=[0, 100])])
        for i in range(0, len(music_name)):
            music_name[i] = music_name[i][:-4]
        cf_ext.data2excel(path='./DB/pkl/name/'+cf_ext.file_list[file][:-4]+'.xlsx',
                          data=result, index_list=music_name)

        # print(cf_ext.cal_grad_melody(entire_music[i]))