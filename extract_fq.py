# 1
# 필요한 module
from __future__ import print_function
import vamp
import os
import librosa
import scipy.stats
import openpyxl
import numpy as np
import pandas as pd
from pandas import DataFrame


# 2-1
# audio 파일 설정
# def audio_setting(path):  # y 값에 load 시키기 전에 경로+음원 이름 설정하기
#     audio_list = []
#     # print(file_list) #음원 이름 확인
#
#     for i in range(len(file_list)):
#         audio_list.append(path + '/' + file_list[i])
#         # print(audio_list)
#
#     return (audio_list)  # 경로+음원 이름
#

class ExtractFrequency:
    def __init__(self):
        self.name_path = []
        self.file_list_name = []
        self.file_name_path=[]

    def folder_name_load(self, folder_path):
        self.folder_list = os.listdir(folder_path)
        print('Folder : ', self.folder_list[0], '~', self.folder_list[-1])  # 폴더 이름 확인
        # print(self.folder_list)
        self.folder_path = []
        for k in range(len(self.folder_list)):
            self.folder_path.append(folder_path + '/' + self.folder_list[k])
            # print(total_path[k])
        #print(self.folder_path)
        # print(self.name_path)
        # print(len(self.name_path))

    def file_name_load(self, folder_path):
        file_list = []
        folder_path=self.folder_path
        for k in range(len(self.folder_list)):
            file_list.append(os.listdir(folder_path[k]))
            #print(file_list[k])
            self.file_name_path.append([])
            self.file_list_name.append([])
            for i in range(len(file_list[k])):
                self.file_name_path[k].append(folder_path[k] + '/' + file_list[k][i])
                self.file_list_name[k].append(file_list[k][i])
        print('file_list_name :: ', self.file_list_name)
        print('folder_path :: ',folder_path)
        print('file_name_path :: ', self.file_name_path)


    # audio 파일 load 하고 default parameter 값을 정한 후 Melodia로 melody 추출
    def load_and_extract_tempo(self, audio_file):  # audio load 하기
        tempo_list = []
        yy = []
        for k in audio_file:
            # tempo 찾기
            audio2, sr = librosa.load(k, sr=44100, mono=True, offset=60, duration=10)
            onset_env = librosa.onset.onset_strength(y=audio2, sr=44100)
            prior = scipy.stats.uniform(30, 300)  # uniform over 30-300 BPM
            utempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, prior=prior)
            utempo
            print(utempo)
            tempo_list.append(utempo)
            # data = vamp.collect(audio, sr=sr, 'mtg-melodia:melodia')
        return (tempo_list)

    def load_and_extract_melody(self, audio_file):  # audio load 하기
        yy = []
        for k in audio_file:
            # print(k) # audio 라는 list 에 잘 들어왔는 지 확인
            audio, sr = librosa.load(k, sr=44100, mono=True)
            # data = vamp.collect(audio, sr=sr, 'mtg-melodia:melodia')
            data = vamp.collect(audio, sample_rate=sr, plugin_key='mtg-melodia:melodia')
            hop, melody = data['vector']
            print(melody)
            print(len(melody))
            # melody 깨끗하게 만들기
            melody_pos = melody[:]
            melody_pos[melody <= 0] = None  # negative value 제거
            print(len(melody_pos))
            yy.append(melody_pos)
        return (yy)


    # 4-1
    # 뽑아낸 melody를 txt로 저장
    def to_pickle(self, title, tempo, melody, i):
        print('title : ', len(title))
        print('tempo : ', len(tempo))
        print('melo : ', len(melody))
        df = pd.DataFrame({"title": title, "tempo": tempo, "melody": melody})
        str = folder_path +self.folder_list[i] + '/freq.pkl'
        df.to_pickle(str, protocol=4)
        print('pickle done : ', i)

if __name__ == '__main__':

    folder_path = 'E:/music_files/팀원들음원이름수정/음원1'
    extF = ExtractFrequency()
    extF.folder_name_load(folder_path)
    extF.file_name_load(extF.folder_path)

    # path='E:/music_files/3'
    # file_list = os.listdir(path)
    # audio_file=audio_setting(path)
    # print(audio_file)

    # 3-1
    for i in range(0,len(extF.file_name_path)):
        tempo_info = extF.load_and_extract_tempo(extF.file_name_path[i])
        melody_info = extF.load_and_extract_melody(extF.file_name_path[i])
        extF.to_pickle(extF.file_list_name[i], tempo_info, melody_info, i)
