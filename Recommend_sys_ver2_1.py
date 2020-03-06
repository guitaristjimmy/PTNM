import pandas as pd
import math
import os
import xlrd
import openpyxl
from matplotlib import font_manager, rc
import numpy as np
import multiprocessing
import time
import pickle


class Recommend_sys:

    def __init__(self, n_path, v_path):
        # self.std = [0.01, 0.0001, 0.001, 0.005, 0.0001, 0.001, 0.001, 0.001]
        self.std = [0.05, 0.005, 0.005, 0.05, 0.005, 0.005, 0.005, 0.005]
        self.v_class = []
        self.name_index = [['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                           'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                           'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.kor_index = ['ㄱ', 'ㄱ', 'ㄴ', 'ㄷ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅂ', 'ㅅ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅈ', 'ㅊ',
                          'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.name_path = self.name_load(base_path=n_path)[0]    # 이름순으로 정렬된 파일들의 경로
        self.val_folder_path = self.name_load(base_path=v_path)[0]     # Value 순으로 정렬된 파일들의 경로
        self.name_table = pd.DataFrame()
        self.name_value = [[], [], [], [], [], []]
        self.sel_name_list = []
        self.weight_table = pd.DataFrame()
        self.territ_data_id_list = []
        self.v_weight_arr = np.array(0)
        self.rank_table = pd.DataFrame()
        self.current_name_list = []
        self.class_territ_list = []
        self.num_class_list = []
        self.territ_list = []
        self.check_table = []

        for i in range(0, 8):
            self.class_territ_list.append([])
            self.num_class_list.append([])

    def input_name(self, name_list):
        estim = []
        for i in range(0, len(name_list)):
            self.current_name_list.append(name_list[i])
            self.sel_name_list.append(name_list[i])
            self.name_search(name=name_list[i])
            estim.append(10)
        self.put_estim_data(song_name_list=name_list, estim_data_list=estim, estim_name='init_weight')

    def name_load(self, base_path):     # y 값에 load 시키기 전에 경로+음원 이름 설정하기
        file_list = os.listdir(base_path)
        # print('file_name : ', file_list[0], '~', file_list[-1])  # 하위 파일 or 폴더 이름 확인
        total_path = []     # 폴더내 모든 파일 경로

        for i in range(len(file_list)):
            total_path.append(base_path + '/' + file_list[i])

        return total_path, file_list

    def data_load(self, path, id_num):

        mem = pd.read_pickle(path)
        if id_num == 0:
            mem = mem.set_index(mem.columns[0])
        else:
            mem = mem.set_index(mem.columns[1])
        return mem

    def name_search(self, name=str):
        l_name = name.lower()
        # print('name :: ', name)
        if self.name_index.count(l_name[0]) != 0:
            temp_table = self.data_load(self.name_path[self.name_index.index(l_name[0])], 0)
        elif l_name[0] in self.name_index[0]:
            temp_table = self.data_load(self.name_path[0], 0)
        else:
            # print(int((ord(name[0])-44032)/588))
            temp_table = self.data_load(self.name_path[self.name_index.index(self.kor_index[int((ord(name[0])-44032)/588)])],
                                        0)
        # print('temp_table :: \n', temp_table)
        name_data = pd.DataFrame(temp_table.loc[name])
        name_data = name_data.T
        # print('name_data :: \n', name_data)
        # print(name_data.index, 'name data load')

        self.name_table_append(name_data)
        self.weight_table_append(list(name_data.index))
        # name_data = list(temp_table.loc[name])
        # for i in range(0, len(self.name_value)):
        #     self.name_value[i].append(name_data[i])

    def name_table_append(self, new_name_table):
        # print(len(self.name_table.index))
        if len(self.name_table.index) != 0:
            self.name_table = self.name_table.append(new_name_table)
        else:
            self.name_table = new_name_table

    def weight_table_append(self, new_id):
        if len(self.weight_table.index) != 0:
            check_list = list(self.weight_table.index)
            for i in range(0, len(new_id)):
                if check_list.count(new_id[i]) != 0:
                    new_id.remove(new_id[i])
            self.weight_table = self.weight_table.append(pd.DataFrame(index=new_id))
        else:
            self.weight_table = pd.DataFrame(index=new_id)

        self.weight_table = self.weight_table.fillna(value=0)

    def territo_data_search(self, center_name, territo, value_id, col_name):
        temp_path = self.name_load(self.val_folder_path[value_id])[0]
        # print('temp path :: ', temp_path)
        # value_name = self.name_table.columns[value_id]
        if int(territo[0]/25) == int(territo[1]/25):
            temp_path_num = int(territo[0]/25)
            if temp_path_num >= 4:
                temp_path_num = 3
            temp_val_table = self.data_load(temp_path[temp_path_num], 1)
        else:
            if int(territo[0]/25) > 3:
                temp_path_num = 3
                temp_val_table = self.data_load(temp_path[temp_path_num], 1)
            else:
                temp_path_num_01 = int(territo[0]/25)
                temp_val_table = self.data_load(temp_path[temp_path_num_01], 1)
                temp_path_num_02 = int(territo[1]/25)
                if temp_path_num_02 >= 4:
                    temp_path_num_02 = 3
                temp_val_table.append(self.data_load(temp_path[temp_path_num_02], 1))
        # print('temp_val_table :: ', temp_val_table)
        val_data = temp_val_table.loc[territo[0]:territo[1]]

        val_data = self.change_value2index(val_data, col_name=col_name)
        return val_data, self.territ_data_id_list

    def change_value2index(self, df, col_name):     #곡제목이 value 값으로 있을때 다시 index로 변환
        dat = np.array(df.index)  # 곡 제목과 변수 정보의 인덱스-벨류 관계를 반전 시키기위해.
        id = np.array(df.values)

        id_list = []
        for k in range(0, len(id)):
            id_list.append(id[k][0])
            self.territ_data_id_list.append(id[k][0])
        return pd.DataFrame(data=dat, index=id_list, columns=col_name)  # 인덱스 - 벨류 위치 바꿈

    def change_index2value(self, df, col_name):     #곡제목을 value 값으로 변환
        dat = np.array(df.index)
        id = np.array(df.values)

        # dat_list = []
        # for k in range(0, len(dat)):
        #     dat_list.append(dat[k][0])
        result = pd.DataFrame(data=dat, index=id, columns=col_name)
        return result

    def item_territo(self, item_value, std_num):
        s = self.std[std_num]
        territo_n = item_value - s
        if territo_n <= 0:
            territo_n = 0
        territo_p = item_value + s

        return [territo_n, territo_p]

    def set_class(self, item_territ, iter_num):
        # print('set_class -------------------------------------------------------------------')
        bar = []
        id, id_r = [], []
        std = self.std[iter_num]*2
        self.v_class.append([])

        for i in range(0, len(item_territ)):
            bar.append(item_territ[i][1])
            id.append(item_territ[i][1])
            id_r.append(item_territ[i][1])

        id_r.reverse()
        # print('id == ', id)
        bar.sort()
        # print('bar == ', bar)

        i = 0
        temp_class = []
        # print('len(bar) : ', len(bar))
        while i < len(bar)-1:
            # print('item_territ :: ', item_territ, 'bar :: ', bar, 'i :: ', i)
            temp_class.append(int(id.index(bar[i])))
            # print('temp_class 01 == ', temp_class)
            for j in range(i+1, len(bar)):
                # print(i, j, 'bar[i] : ', bar[i], 'bar[j] : ', bar[j])
                if (bar[i]-std <= bar[j]) and (bar[i]+std >= bar[j]):
                    if bar.count(bar[j]) == 1:
                        temp_class.append(int(id.index(bar[j])))
                        # print('temp_class 02 == ', temp_class)
                    else:
                        temp_class.append(len(bar)-1 - int(id_r.index(bar[j])))
                    if j == len(bar)-1:
                        i = len(bar)
                elif len(temp_class) != 1:
                    i = j-1
                    break
                else:
                    i = j
                    break

                if j == len(bar)-1:
                    i += 1

            # print('len_TempClass :: ', len(temp_class), '///  i :: ', i, '///   iter_num :: ', iter_num)
            if len(temp_class) >= 2:
                if (len(self.v_class[iter_num]) == 0) or (self.v_class[iter_num][-1].count(temp_class[0]) == 0):
                    temp_class.sort()
                    self.v_class[iter_num].append(temp_class)
                else:
                    # print('temp_class[1:] :: ', temp_class[1:])
                    for k in range(0, len(temp_class[1:])):
                        self.v_class[iter_num][-1].append(temp_class[1:][k])
                    self.v_class[iter_num][-1].sort()
            temp_class = []
        # print('v_class :: ', self.v_class)

    def class_territ(self, indiv_class, t_list, iter_num):
        in_territ = []
        # print('indiv_class :: ', indiv_class)
        # print('t_list :: ', t_list)
        # print('check :: \n t_list :: ', t_list, '\n indiv_class :: ', indiv_class)
        for i in range(0, len(indiv_class)):
            in_territ.append(t_list[iter_num][indiv_class[i]][0])
            in_territ.append(t_list[iter_num][indiv_class[i]][1])
        in_territ.sort()
        # print('temp_territo :: ', temp)
        # print('check v_class:: ', sel
        # f.v_class)
        if len(self.class_territ_list[iter_num]) == 0:
            self.class_territ_list[iter_num].append([in_territ[0], in_territ[-1]])
            self.num_class_list[iter_num].append(len(indiv_class))
        else:
            sum_flag = 0
            temp_num_class = len(indiv_class)
            remove_list = []
            remove_num_list = []
            for j in range(0, len(self.class_territ_list[iter_num])):
                # print('class_territ_list[iter_num][j][0] :: ', self.class_territ_list[iter_num][j][0])
                # print('temp :: ', temp)
                if in_territ[0] < self.class_territ_list[iter_num][j][0] < in_territ[-1]:

                    if sum_flag == 0:
                        self.num_class_list[iter_num][j] += temp_num_class
                        temp_num_class = self.num_class_list[iter_num][j]
                        sum_flag = 1

                        if self.class_territ_list[iter_num][j][1] < in_territ[-1]:
                            self.class_territ_list[iter_num][j] = [in_territ[0], in_territ[-1]]
                        else:
                            self.class_territ_list[iter_num][j] = [in_territ[0], self.class_territ_list[iter_num][j][1]]
                            in_territ = [in_territ[0], self.class_territ_list[iter_num][j][1]]
                    else:
                        # temp = temp_num_class
                        self.num_class_list[iter_num][j] += temp_num_class
                        temp_num_class = self.num_class_list[iter_num][j]
                        remove_num_list.append(self.num_class_list[iter_num][j-1])

                        remove_list.append(self.class_territ_list[iter_num][j - 1])
                        if self.class_territ_list[iter_num][j][1] < in_territ[-1]:
                            self.class_territ_list[iter_num][j] = [in_territ[0], in_territ[-1]]
                        else:
                            self.class_territ_list[iter_num][j] = [in_territ[0], self.class_territ_list[iter_num][j][1]]
                            in_territ = [in_territ[0], self.class_territ_list[iter_num][j][1]]

                elif in_territ[0] < self.class_territ_list[iter_num][j][1] < in_territ[-1]:
                    if sum_flag == 0:
                        self.num_class_list[iter_num][j] += temp_num_class
                        temp_num_class = self.num_class_list[iter_num][j]
                        sum_flag = 1

                        if self.class_territ_list[iter_num][j][0] > in_territ[0]:
                            self.class_territ_list[iter_num][j] = [in_territ[0], in_territ[-1]]
                        else:
                            self.class_territ_list[iter_num][j] = [self.class_territ_list[iter_num][j][0], in_territ[-1]]
                            in_territ = [self.class_territ_list[iter_num][j][0], in_territ[-1]]

                    else:
                        # in_territ = temp_num_class
                        self.num_class_list[iter_num][j] += temp_num_class
                        temp_num_class = self.num_class_list[iter_num][j]
                        remove_num_list.append(self.num_class_list[iter_num][j-1])

                        remove_list.append(self.class_territ_list[iter_num][j - 1])
                        if self.class_territ_list[iter_num][j][0] > in_territ[0]:
                            self.class_territ_list[iter_num][j] = [in_territ[0], in_territ[-1]]
                        else:
                            self.class_territ_list[iter_num][j] = [self.class_territ_list[iter_num][j][0], in_territ[-1]]
                            in_territ = [self.class_territ_list[iter_num][j][0], in_territ[-1]]

                elif (in_territ[-1] < self.class_territ_list[iter_num][j][1]) and \
                     (in_territ[0] > self.class_territ_list[iter_num][j][0]):

                    if sum_flag == 0:
                        self.num_class_list[iter_num][j] += temp_num_class
                        temp_num_class = self.num_class_list[iter_num][j]
                        sum_flag = 1
                        in_territ = [self.class_territ_list[iter_num][j][0], self.class_territ_list[iter_num][j][0]]

                    else:
                        # temp = temp_num_class
                        self.num_class_list[iter_num][j] += temp_num_class
                        temp_num_class = self.num_class_list[iter_num][j]
                        remove_num_list.append(self.num_class_list[iter_num][j-1])
                        remove_list.append(self.class_territ_list[iter_num][j - 1])

                        in_territ = [self.class_territ_list[iter_num][j][0], self.class_territ_list[iter_num][j][0]]

                else:
                    # print('in_territ :: ', in_territ, '\n class_territ_list[iter_num ::', self.class_territ_list[iter_num])
                    sum_flag = 0
            for j in range(0, len(remove_num_list)):
                # print('num_class_list[iter_num] :: ', self.num_class_list)
                # print('remove_num_list[j] :: ', remove_num_list[j])
                self.num_class_list[iter_num].remove(remove_num_list[j])
                self.class_territ_list[iter_num].remove(remove_list[j])
            if temp_num_class == len(indiv_class):
                self.class_territ_list[iter_num].append([in_territ[0],in_territ[-1]])
                self.num_class_list[iter_num].append(len(indiv_class))
        # print('class_territ :: ', self.class_territ_list)
        # print('v_class :: ', self.v_class)
        # print('num_class_list :: ', self.num_class_list)
        return self.class_territ_list[iter_num]

    def cal_gaussian_weight(self, x, u, i_w, std_num):
        std = self.std[std_num]
        # print('x-u**2 ::', 2*math.exp(-((x-u)**2)/(2*std)))
        g_weight = i_w*1.5*(math.exp(-(((x-u)**2) / (2*std))))
        # 0.1655
        return g_weight

    def cal_class_weight(self, class_territory, num_class, mean_estim):
        if len(class_territory) == 0:
            return 0
        else:
            # print('check :: ', class_territory)
            class_weight = ((num_class**2)*mean_estim)/(class_territory[1]-class_territory[0])
            # print('class_weight :: ', class_territory)
            return class_weight

    def cal_val_weight(self, class_weight_list):
        val_weight = []
        for i in range(0, len(class_weight_list)):
            if len(class_weight_list[i]) != 0:
                val_weight.append(sum(class_weight_list[i])/(100+len(class_weight_list[i])))
            else:
                val_weight.append(0)
        vw_arr = np.array(val_weight)
        self.v_weight_arr = vw_arr.reshape((8, 1))

    def update_nametable(self, df):
        song_name = list(df.index)

        # 이미 name_table에 추가한 이름을 찾아낸다.
        check_list = list(self.name_table.index)
        s = time.time()
        for i in range(0, len(check_list)):
            if song_name.count(check_list[i]) != 0:
                song_name.remove(check_list[i])
        e = time.time()
        print('time check list :: ', e-s)

        s = time.time()

        # length_01 = (0, int(len(song_name)/4))
        # length_02 = (int(len(song_name)/4), int(len(song_name)/2))
        # length_03 = (int(len(song_name)/2), int(len(song_name)*3/4))
        # length_04 = (int(len(song_name)*3/4), len(song_name))
        #
        # p_01 = super().Process(target=self.update_nametable_thread, args=(song_name, length_01))
        # p_02 = super().Process(target=self.update_nametable_thread, args=(song_name, length_02))
        # p_03 = super().Process(target=self.update_nametable_thread, args=(song_name, length_03))
        # p_04 = super().Process(target=self.update_nametable_thread, args=(song_name, length_04))
        #
        # p_01.start()
        # p_02.start()
        # p_03.start()
        # p_04.start()
        #
        # p_01.join()
        # p_02.join()
        # p_03.join()
        # p_04.join()
        for i in range(0, len(song_name)):
            self.name_search(song_name[i])
        # self.name_table = self.name_table.join(df)
        # self.name_table = self.name_table.drop_duplicates(keep='first')
        e = time.time()
        print('time song name search :: ', e-s)

    def update_nametable_thread(self, song, length):
        for i in range(length[0], length[1]):
            self.name_search(song[i])

    def join_weighttable(self, df):
        # song_name = np.array(df.index)
        self.weight_table = self.weight_table.join(df)
        self.weight_table = self.weight_table.fillna(value=0)

    def update_weighttable(self, df):
        self.weight_table.update(df)

    def put_estim_data(self, song_name_list, estim_name, estim_data_list):
        # print('estim_data_list :: ', estim_data_list)
        # print('song_name_list :: ', song_name_list)

        estim_frame = pd.DataFrame(data=estim_data_list, index=song_name_list)
        estim_frame.columns = [estim_name]
        # print('estim_frame :: ', estim_frame)

        if len(self.weight_table.index) == 0:
            self.weight_table = estim_frame
        else:
            self.weight_table = self.weight_table.join(estim_frame)

    def put_class_weight(self, cw_list, c_territ):
        temp_list = []
        for i in range(0, len(self.name_table.columns)):

            c_w_name = str(self.name_table.columns[i]) + '_C' + str(0) + '_w'       # Class Weight 이름
            if len(c_territ[i]) == 0:       # class가 형성되지 않으면 0을 입력해 둔다.
                temp = pd.DataFrame(index=self.weight_table.index)
                temp[c_w_name] = 0
                check = list(self.weight_table.columns)
                if check.count(c_w_name) == 0:
                    self.join_weighttable(temp)
                else:
                    self.update_weighttable(temp)
            else:       # class weight 값을 입력

                for j in range(0, len(c_territ[i])):
                    temp = self.name_table[self.name_table.columns[i]]
                    temp = self.change_index2value(df=temp, col_name=['Name'])
                    temp = temp.sort_index()
                    # print('c_territ[i][j] :: ', c_territ[i][j][0], c_territ[i][j][1])
                    temp = temp.loc[c_territ[i][j][0]:c_territ[i][j][1]]
                    # print('c_territ :: ', c_territ)
                    # print('check :: ', i, j, '\n', cw_list)
                    temp[c_w_name] = cw_list[i][j]
                    temp = temp.set_index('Name')
                    check = list(self.weight_table.columns)
                    if check.count(c_w_name) == 0:
                        self.join_weighttable(temp)
                    else:
                        self.update_weighttable(temp)

    def cal_rank(self):

        # print('weight_table :: \n', self.weight_table.head(10))
        territ_w = self.weight_table.columns[1:9]
        class_w = self.weight_table.columns[9:]
        territ_w = np.array(self.weight_table[territ_w])
        class_w = np.array(self.weight_table[class_w])
        rank_df = pd.DataFrame(data=np.dot((territ_w*class_w), self.v_weight_arr), index=self.weight_table.index)
        rank_df = rank_df.rank(ascending=False)
        rank_df.columns = ['Total_Weight']
        self.rank_table = rank_df.sort_values(by='Total_Weight')

    def recommand_song(self):
        recom = self.rank_table
        for i in range(0 , len(self.sel_name_list)):
            recom = recom.drop(index=self.sel_name_list[i])
        recom = recom.iloc[0:5]
        recom = list(recom.index)
        for i in range(0, len(recom)):
            self.sel_name_list.append(recom[i])

        self.current_name_list = recom

        return recom

    def update_feedback(self, feedback_data, song_name):
        temp = pd.DataFrame({'init_weight' : feedback_data}, index=song_name)
        # print('temp :: \n', temp)
        self.weight_table.update(temp)

    def cal_territ_weight(self):
        # Determine Song Territory -----------------------------------------------
        # print('# of value :: ', len(recom.name_table.columns))
        # print(recom.name_table['Chroma'][0])
        self.v_class = []       # v_class 초기화
        self.territ_list = []   # self.territ_list 초기화
        result_id = []
        result_table = []
        for i in range(0, len(self.name_table.columns)):       # 변수 개수만큼 반복
            self.territ_list.append([])      # 곡의 범위를 담는다.
            i_name = self.name_table.columns[i]        # i_name :: value Name
            result_table.append([])     # 결과를 임시 저장한다.
            for j in range(0, len(self.current_name_list)):
                # 곡의 영역 정보 저장
                # print('name_table :: ', self.name_table[i_name][self.current_name_list[j]])
                self.territ_list[i].append(self.item_territo(self.name_table[i_name][self.current_name_list[j]], i))
                # print('territ_list[i] :: ', self.territ_list[i])
                song_weight, result_id = self.territo_data_search(center_name=self.current_name_list[j],
                                                                  territo=self.territ_list[i][j], value_id=i,
                                                                  col_name=[i_name])
                                                                    # 영역 내부 곡 정보 로드
                # print('song_weight :: \n', song_weight)
                # print('id_list :: ', id_list)
                w_sg = []
                for k in range(0, len(song_weight[i_name].values)):
                    # print('x :: ', song_weight.iloc[k],
                    #       '\n u ::', self.name_table.loc[self.current_name_list[j]][i_name])
                    # print('gaussian weight :: ', self.cal_gaussian_weight(x=float(song_weight[i_name].values[k]),
                    #                                      u=self.name_table.loc[self.current_name_list[j]][i_name],
                    #                                      std_num=i))

                    w_sg.append(self.cal_gaussian_weight(x=float(song_weight[i_name].values[k]),
                                                         u=self.name_table.loc[self.current_name_list[j]][i_name],
                                                         i_w=self.weight_table.loc[self.current_name_list[j]]['init_weight'],
                                                         std_num=i))
                    # 각 곡당의 weight 계산.
                # print('w_sg :: ', w_sg)
                song_weight[i_name + '_(w_sg)'] = w_sg
                result_table[i].append(song_weight)     # result table에는 각 변수별로 각 곡의 weight 정보를 dataframe으로 담는다.
                # print('i :: ', i)
                # print('result table[i] :: \n', result_table[i])
            self.set_class(self.territ_list[i], i)      # 변수 축마다 영역 정보를 바탕으로 클래스 형성

        # territ_list 완성 시점 // song_weight 정보 완성

        # print('result_table :: \n', result_table)
        result_shape = pd.DataFrame(data=result_id, columns=['name'])       # result_shape은 말그대로 최종 정보를 담기위한 틀이다.
        result_shape = result_shape.drop_duplicates('name', keep='first')   # 곡의 이름정보만 갖고있고 빈 틀이다.
        result_shape = result_shape.set_index('name')
        result = result_shape
        result_list = []
        self.territ_data_id_list = []      # result_id입력 후 초기화
        for i in range(0, len(result_table)):
            temp_02 = pd.DataFrame()
            for j in range(0, len(result_table[i])):
                temp = pd.merge(left=result_shape, right=result_table[i][j], left_index=True, right_index=True, how='outer')
                temp = temp.fillna(value=0)
                if len(temp_02.index) == 0:
                    temp_02 = temp
                else:
                    temp_02 = temp_02 + temp
            result = result.join(temp_02)
        # print('result table :: ', result_table)
        feature_name = ['Chroma', 'sBW', 'MelSpecto', 'sCentroid', 'sContrast', 'Tonnetz', 'Code_Feature', 'MFCC']
        return result.drop(columns=feature_name)

    def set_class_var_weight(self):
        c_territ = []
        num_class = []
        estim_mean = []
        c_w = []
        # print('v_class :: ', self.v_class)
        for i in range(0, len(self.name_table.columns)):
            num_class.append([])
            estim_mean.append([])
            c_w.append([])
            # if len(self.v_class[i]) == 0:
            #     num_class[i].append(0)
            #     # estim_mean[i].append(0)
            #     c_w[i].append(0)
            # else:
            for j in range(0, len(self.v_class[i])):
                self.class_territ(self.v_class[i][j], self.territ_list, i)
                num_class[i].append(len(self.v_class[i][j]))
                et_sum = 0
                # print('v_class :: ', self.v_class)
                # print('class_territ_list :: ', self.class_territ_list)
                # print('num_class_list[i][j] :: ', self.num_class_list[i][j])
            for j in range(0, len(self.class_territ_list[i])):

                # print('check :: ', self.name_table.columns[i])
                temp_01 = self.name_table[self.name_table.columns[i]] < self.class_territ_list[i][j][1]
                temp_01 = self.name_table[temp_01]
                # print('temp_01 :: ', temp_01)
                temp_02 = temp_01[self.name_table.columns[i]] > self.class_territ_list[i][j][0]
                temp_02 = temp_01[temp_02]
                # print('temp_02 :: ', temp_02)

                if len(temp_02.index) == 0:
                    estim_mean[i].append(0)
                else:
                    estim_df = self.weight_table.loc[temp_02.index]
                    estim_df = pd.DataFrame(estim_df['init_weight'], index=estim_df.index)
                    # print('estim_df :: \n', estim_df)
                    # print('estim_df.mean()', estim_df.mean(axis=0))
                    estim_m = float(estim_df.mean())
                    estim_mean[i].append(estim_m)
                # print('estim_mean[i][j] :: ', estim_mean[i][j])
                # print('class_territ_list[i][j] :: ', self.class_territ_list[i][j])
                c_w[i].append(self.cal_class_weight(class_territory=self.class_territ_list[i][j],
                                                    num_class=self.num_class_list[i][j],
                                                    mean_estim=estim_mean[i][j]))
        # print('v_class :: ', self.v_class)
        # print('c_territ :: ', self.class_territ_list)
        # print('c_w :: ', c_w)
        self.put_class_weight(cw_list=c_w, c_territ=self.class_territ_list)
        self.cal_val_weight(c_w)

class PTNM_Recommend(Recommend_sys):
    def __init__(self, call_list):
        super().__init__(n_path='D:/Python_Note/Project/Kisung/PTNM_Feature_Extract/DB/name_pickle',
                          v_path='D:/Python_Note/Project/Kisung/PTNM_Feature_Extract/DB/value_pickle')
        self.input_songs = call_list
        super().input_name(name_list=self.input_songs)
        self.end_flag = 0
        self.iter_count = 0

    def start_recom(self):

        # print('name_table :: \n', recom.name_table)
        # print('name_value :: ', recom.name_value)

        # Determine Song Territory -------------------------------------------------------------------------------------
        # Set Class ----------------------------------------------------------------------------------------------------
        # print('v_class :: ', recom.v_class)
        s = time.time()
        territ_weight = super().cal_territ_weight()
        e = time.time()
        print('time territ_weight :: ', e-s)
        # Update NameTable ---------------------------------------------------------------------------------------------
        s = time.time()
        super().update_nametable(territ_weight)
        e = time.time()
        print('time update_name_table :: ', e-s)
        s = time.time()
        super().join_weighttable(territ_weight)
        e = time.time()
        print('time update_weight_table :: ', e-s)
        # Calculate & Set Class/Variable Weight ------------------------------------------------------------------------
        s = time.time()
        super().set_class_var_weight()
        e = time.time()
        print('time set class,var weight :: ', e-s)
        # print('v_w :: ', recom.v_weight_arr)

        # Calculate Rank -----------------------------------------------------------------------------------------------
        s = time.time()
        super().cal_rank()
        e = time.time()
        print('time cal rank :: ', e-s)
        recom_song_list = super().recommand_song()

        feedback = []
        self.iter_count += 1
        print('PTNM 추천 top 5 ::', self.iter_count, '번째 추천!!')
        print(recom_song_list)
        for i in range(0, len(recom_song_list)):
            feedback.append(float(input(recom_song_list[i]+'의 평가를 입력하세요 :: ')))
        # print('feedback :: ', feedback)

        super().update_feedback(feedback_data=feedback, song_name=recom_song_list)

    def recom_steps(self):
        # Determine Song Territory -------------------------------------------------------------------------------------
        # Set Class ----------------------------------------------------------------------------------------------------
        # print('v_class :: ', recom.v_class)

        s = time.time()
        territ_weight = super().cal_territ_weight()
        e = time.time()
        print('time territ_weight :: ', e-s)
        # Update NameTable ---------------------------------------------------------------------------------------------
        s = time.time()
        super().update_nametable(territ_weight)
        e = time.time()
        print('time update_name_table :: ', e-s)
        s = time.time()
        super().update_weighttable(territ_weight)
        e = time.time()
        print('time update_weight_table :: ', e-s)

        # Calculate & Set Class/Variable Weight ------------------------------------------------------------------------
        s = time.time()
        super().set_class_var_weight()
        e = time.time()
        print('time set class,var weight :: ', e-s)
        # print('v_w :: ', self.v_weight_arr)

        # Calculate Rank -----------------------------------------------------------------------------------------------
        s = time.time()
        super().cal_rank()
        e = time.time()
        print('time set class,var weight :: ', e-s)
        recom_song_list = super().recommand_song()

        feedback = []

        self.iter_count += 1
        print('PTNM 추천 top 5 ::', self.iter_count, '번째 추천!!')
        print(recom_song_list)
        print('\t\t\t\t 종료키는 \'120\' 입니다.\n')
        for i in range(0, len(recom_song_list)):
            input_feedback = float(input(recom_song_list[i]+'의 평가를 입력하세요 :: '))
            if input_feedback != 120:
                feedback.append(input_feedback)
            else:
                print('-----:: 추천을 종료합니다. ::-----')
                self.end_flag = 1
                return 0
        # print('feedback :: ', feedback)

        super().update_feedback(feedback_data=feedback, song_name=recom_song_list)


if __name__ == '__main__':

    # path = 'D:/Python_Note/Project/Kisung/PTNM_Feature_Extract/Test.xlsx'
    # item = '전인권 - 걱정말아요 그대.mp3'

    call_list = ['레드핫칠리페퍼스-DaniCalifornia', '레드핫칠리페퍼스-TellMeBaby', '레드핫칠리페퍼스-Snow__Hey_Oh']

    recom = PTNM_Recommend(call_list=call_list)

    recom.start_recom()

    while recom.end_flag == 0:
        recom.recom_steps()
