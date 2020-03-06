from __future__ import print_function
from __future__ import division
import os
import numpy as np
import pandas as pd
import cv2
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


class HistoComparison(Arithmet):

    def __init__(self):
        super().__init__()
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

    def name_load(self, folder_path):  # y 값에 load 시키기 전에 경로+음원 이름 설정하기
        self.folder_list = os.listdir(folder_path)
        print('Folder : ', self.folder_list[0], '~', self.folder_list[-1])  # 폴더 이름 확인
        total_path = []
        file_list = []
        file_list_name = []
        name_path = []
        for k in range(len(self.folder_list)):
            total_path.append(folder_path + '/' + self.folder_list[k])
            file_list.append(os.listdir(total_path[k]))
            # print(total_path[k])
            print(file_list[k])
            for i in range(len(file_list[k])):
                name_path.append(total_path[k] + '/' + file_list[k][i])
                file_list_name.append(file_list[k][i])
        # print(name_path)
        # print(file_list_name)
        return (name_path, file_list_name)

    def cal_img_rank(self, img_list):
        img_mean_list = []
        for i in range(0, len(img_list)):
            img_mean_list.append(img_list[i].mean())
        rank = pd.Series(img_mean_list)
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

    def sContra_crop(self, img):
        crop = [0, 0, 0, 0, 0, 0, 0]
        crop[0] = img[56:311, 77:2550]
        crop[1] = img[317:575, 77:2250]
        crop[2] = img[581:839, 77:2250]
        crop[3] = img[845:1103, 77:2250]
        crop[4] = img[1109:1367, 77:2250]
        crop[5] = img[1373:1631, 77:2250]
        crop[6] = img[1637:1892, 77:2250]
        return crop

    def ChromaHist(self, group_crop_img, target_crop_img, g_size, target_index,process_type=str):
        group_cr_rank=[]
        print(group_crop_img[0])
        for i in range(0, len(group_crop_img)):
            group_cr_rank.append(self.cal_img_rank(group_crop_img[i]))
        print(group_cr_rank)
    # Convert them to HSV format:
        hsv_group = []
        hsv_target = []

    # hist list만들기
        hist_group = []
        hist_target = []
        baseVStest = []
    # group_hsv_hist기본+normalize까지
        for g in range(len(group_crop_img)):
            hsv_group.append([])
            hist_group.append([])
            for l in range(0, 12):
                hsv_group[g].append(cv2.cvtColor(group_crop_img[g][l], cv2.COLOR_BGR2HSV))
                hist_group[g].append(cv2.calcHist([hsv_group[g][l]],
                                                self.channels, None, self.histSize, self.ranges, accumulate=False))
                cv2.normalize(hist_group[g][l], hist_group[g][l], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # target_hsv_hist기본+normalize까지
        for t in range(len(target_crop_img)):
            hsv_target.append([])
            hist_target.append([])
            baseVStest.append([])
            for l in range(0, 12):
                hsv_target[t].append(cv2.cvtColor(target_crop_img[t][l], cv2.COLOR_BGR2HSV))
                hist_target[t].append(cv2.calcHist([hsv_target[t][l]],
                                                self.channels, None, self.histSize, self.ranges, accumulate=False))
                cv2.normalize(hist_target[t][l], hist_target[t][l], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                # print(hist_test)


    #group과 target 비교 및 avg-minMAX적용+그룹별통합
        baseVAStest=[]
        for t in range(len(target_crop_img)):
            baseVAStest.append([])
            for g in range(len(group_crop_img)):
                for l in range(0, 12):
                    # (compare_Method=  ALT_Chi-square = 4)
                    compare_hist = cv2.compareHist(hist_group[g][l], hist_target[t][l], 4)
                    compare_weight = list(compare_hist * group_cr_rank[g])
                baseVStest[t].append(self.avg_minMAX(compare_weight))
                if (g % g_size) == 4:
                    baseVAStest[t].append(5 / (baseVStest[t][g - 4] + baseVStest[t][g - 3] +
                                             baseVStest[t][g - 2] + baseVStest[t][g - 1] + baseVStest[t][g]))
                    # 전체 값 0~1값으로 normalization
                baseVAStest[t] = super().normalize_data(baseVAStest[t])
        # #상속받아 arithmet적용
            super().cal_cumul_p(baseVAStest[t])
            baseVAStest[t] = [(super().arith([0, 100]))]
            self.cumul_p = []
    # Chroma 엑셀 저장할 위치 및 이름 설정해주기
        #print('data == ', baseVAStest)
        pyo = self.dat2excel(process_type, data=baseVAStest, id=target_index)
        return pyo

    def tonnetz_Contra_Hist(self, group_crop_img, target_crop_img, g_size, target_index, process_type=str):

        if (process_type =='Tonnetz'):
            linesCount = 6
        elif (process_type == 'Contra'):
            linesCount = 7

    # Convert them to HSV format:
        hsv_group = []
        hsv_target = []

    # hist list만들기
        hist_group = []
        hist_target = []
        baseVStest = []

    # group_hsv_hist기본+normalize까지
        for g in range(len(group_crop_img)):
            hsv_group.append([])
            hist_group.append([])
            for l in range(0, linesCount):
                hsv_group[g].append(cv2.cvtColor(group_crop_img[g][l], cv2.COLOR_BGR2HSV))
                hist_group[g].append(cv2.calcHist([hsv_group[g][l]],
                                                self.channels, None, self.histSize, self.ranges, accumulate=False))
                cv2.normalize(hist_group[g][l], hist_group[g][l], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # target_hsv_hist기본+normalize까지
        for t in range(len(target_crop_img)):
            hsv_target.append([])
            hist_target.append([])
            baseVStest.append([])
            for l in range(0, linesCount):
                hsv_target[t].append(cv2.cvtColor(target_crop_img[t][l], cv2.COLOR_BGR2HSV))
                hist_target[t].append(cv2.calcHist([hsv_target[t][l]],
                                                self.channels, None, self.histSize, self.ranges, accumulate=False))
                cv2.normalize(hist_target[t][l], hist_target[t][l], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                # print(hist_test)
    # group과 target 비교 및 sumsq적용+그룹별통합
        baseVAStest=[]
        for t in range(len(target_crop_img)):
            baseVAStest.append([])
            for g in range(len(group_crop_img)):
                sumsq_base = []
                for l in range(0, linesCount):
                    # (compare_Method=Chi-square=1)
                    sumsq_base.append(cv2.compareHist(hist_group[g][l], hist_target[t][l], 1))
                baseVStest[t].append(self.sumsq(sumsq_base))
                if (g % g_size) == 4:
                    baseVAStest[t].append(5 / (baseVStest[t][g - 4] + baseVStest[t][g - 3] +
                                               baseVStest[t][g - 2] + baseVStest[t][g - 1] + baseVStest[t][g]))
                    # 전체 값 0~1값으로 normalization
            baseVAStest[t]=super().normalize_data(baseVAStest[t])
        #상속받아 arithmet적용
            super().cal_cumul_p(baseVAStest[t])
            baseVAStest[t] = [(super().arith([0, 100]))]
            self.cumul_p = []

    # 토넷츠 엑셀 저장할 위치 및 이름 설정해주기
        pyo = self.dat2excel(process_type, data=baseVAStest, id=target_index)
        return pyo

    def histo_compare(self, group_img, target_img, g_size, target_index, process_type=str):
        # Convert them to HSV format:
        hsv_group = []
        hsv_target = []

        # hist list만들기
        hist_group = []
        hist_target = []

        for g in range(len(group_img)):
            hsv_group.append(cv2.cvtColor(group_img[g], cv2.COLOR_BGR2HSV))
            hist_group.append(cv2.calcHist([hsv_group[g]], self.channels, None,
                                              self.histSize, self.ranges, accumulate=False))
            cv2.normalize(hist_group[g], hist_group[g], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        for t in range(len(target_img)):
            hsv_target.append(cv2.cvtColor(target_img[t], cv2.COLOR_BGR2HSV))
            hist_target.append(cv2.calcHist([hsv_target[t]], self.channels, None,
                                               self.histSize, self.ranges, accumulate=False))
            cv2.normalize(hist_target[t], hist_target[t], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        compare_data = []
        result = []
        for t in range(0, len(target_img)):
            compare_data_base = []
            compare_data.append([])
            for g in range(0, len(group_img)):
                compare_data_base.append(cv2.compareHist(hist_group[g], hist_target[t], 1))
                if (g % g_size) == 4:
                    compare_data[t].append(5 / (compare_data_base[g - 4] + compare_data_base[g - 3] +
                                                compare_data_base[g - 2] + compare_data_base[g - 1] +
                                                compare_data_base[g]))
    # #전체 값 0~1값으로 normalization
            data = super().normalize_data(compare_data[t])
            super().cal_cumul_p(data)
            result.append([super().arith([0, 100])])
            self.cumul_p = []

        pyo = self.dat2excel(process_type, data=result, id=target_index)
        return pyo

    def avg_minMAX(self, n):
        #'행'형태로 받을 것 ex)n=a[i]
        return (sum(n)-min(n)-max(n))/10

    def sumsq(self, n):
        # calculate_Squre_sum
        s = 0
        for i in n:
            s = s + i**2
        return s/6

    def dat2excel(self, process_type, data, id):
        save_name = 'C:/Users/s/Desktop/03_05_EDIYA' + process_type + '.xlsx'
        compare = pd.DataFrame.from_records(data, index=id)
        compare = compare.sort_values(by=[0],ascending=[True])
        compare.to_excel(save_name)
        return compare

    def readKorean(self, path):
        src_base = open(path, "rb")
        bytes = bytearray(src_base.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        return bgrImage

if __name__ == '__main__':
    ext = HistoComparison()
##group-경로설정
# chroma용 path설정
    ch_group_folder_path = 'C:/Users/s/Desktop/PTNM_Feature_Analysis/Group/group_Chroma'
    chroma_group_name_path, ch_group_fileList_name = ext.name_load(ch_group_folder_path)
    # group_name_ch=[ext.folder_list]
# tonnetz용 path설정
    to_group_folder_path = 'C:/Users/s/Desktop/PTNM_Feature_Analysis/Group/group_Tonnetz'
    tonnetz_group_name_path, to_group_fileList_name = ext.name_load(to_group_folder_path)
    # group_name_to=[ext.folder_list]
# sBW용 path설정
    BW_group_folder_path = 'C:/Users/s/Desktop/PTNM_Feature_Analysis/Group/group_sBW'
    BW_group_name_path, BW_group_fileList_name = ext.name_load(BW_group_folder_path)
    # group_name_sBW=[ext.folder_list]
# sCentroid용 path설정
    Cen_group_folder_path = 'C:/Users/s/Desktop/PTNM_Feature_Analysis/Group/group_sCentroid'
    Cen_group_name_path,Cen_group_fileList_name = ext.name_load(Cen_group_folder_path)
    # group_name_Cen=[ext.folder_list]
# # sContra용 path설정
    Ctra_group_folder_path = 'C:/Users/s/Desktop/PTNM_Feature_Analysis/Group/group_sContra'
    Ctra_group_name_path, Ctra_group_fileList_name = ext.name_load(Ctra_group_folder_path)
    # group_name_Ctra=[ext.folder_list]
# MelSpecto용 path설정
    MelS_group_folder_path = 'C:/Users/s/Desktop/PTNM_Feature_Analysis/Group/group_MelSpecto'
    MelS_group_name_path, MelS_group_fileList_name = ext.name_load(MelS_group_folder_path)
    # group_name_MelS=[ext.folder_list]

#Target-경로설정
    ch_target_folder_path = 'C:/Users/s/Desktop/PTNM_Feature_Analysis/Target/Feature_Img_2020-02-13/target_Chroma'
    ch_target_name_path, ch_target_fileList_name = ext.name_load(ch_target_folder_path)

    to_target_folder_path = 'C:/Users/s/Desktop/PTNM_Feature_Analysis/Target/Feature_Img_2020-02-13/target_Tonnetz'
    to_target_name_path, to_target_fileList_name=ext.name_load(to_target_folder_path)

    BW_target_folder_path = 'C:/Users/s/Desktop/PTNM_Feature_Analysis/Target/Feature_Img_2020-02-13/target_sBW'
    BW_target_name_path, BW_target_fileList_name = ext.name_load(BW_target_folder_path)

    Cen_target_folder_path = 'C:/Users/s/Desktop/PTNM_Feature_Analysis/Target/Feature_Img_2020-02-13/target_sCentroid'
    Cen_target_name_path, Cen_target_fileList_name = ext.name_load(Cen_target_folder_path)

    Ctra_target_folder_path = 'C:/Users/s/Desktop/PTNM_Feature_Analysis/Target/Feature_Img_2020-02-13/target_sContra'
    Ctra_target_name_path, Ctra_target_fileList_name = ext.name_load(Ctra_target_folder_path)

    MelS_target_folder_path = 'C:/Users/s/Desktop/PTNM_Feature_Analysis/Target/Feature_Img_2020-02-13/target_MelSpecto'
    MelS_target_name_path, MelS_target_fileList_name = ext.name_load(MelS_target_folder_path)

#crop하는 애들만
    group_chroma_crop_img = []
    target_chroma_crop_img = []
    group_tonnetz_crop_img = []
    target_tonnetz_crop_img = []
    group_Ctra_crop_img = []
    target_Ctra_crop_img = []
#index찍어주기위한 target의 이름들
    target_name = []
#각 group에 5개씩 이미지가 들어있음을 나타내는 줄
    group_size = 5

    ch_group_num = 8
    to_group_num = 11
    sBW_group_num = 7
    sCentroid_group_num = 4
    sCtra_group_num = 6
    MelSpecto_group_num = 5


##group####
##crop안하는애들은 []생성후append, crop하면cr로 옮기고 group.img에 append
# # group__chroma
    for g in range(0, len(chroma_group_name_path)):
        cr = ext.readKorean(chroma_group_name_path[g])
        group_chroma_crop_img.append(ext.chroma_crop(cr))
        print('group_C_cropping done-',g)
# group__tonnetz
    for g in range(0, len(tonnetz_group_name_path)):
        t_cr = ext.readKorean(tonnetz_group_name_path[g])
        group_tonnetz_crop_img.append(ext.tonnetz_crop(t_cr))
        print('group_T_cropping done-',g)
#group_sBW
    group_BW=[]
    for g in range(0, len(BW_group_name_path)):
        group_BW.append(ext.readKorean(BW_group_name_path[g]))
        print('group_sBW_PreProcessing done-', g)
# group__sCentroid
    group_Centroid=[]
    for g in range(0, len(Cen_group_name_path)):
        group_Centroid.append(ext.readKorean(Cen_group_name_path[g]))
        print('group_sCentroid_PreProcessing done-', g)
#group__sContra
    for g in range(0, len(Ctra_group_name_path)):
        Ctra_cr=ext.readKorean(Ctra_group_name_path[g])
        group_Ctra_crop_img.append(ext.sContra_crop(Ctra_cr))
        print('group_sContra_cropping done-', g)
# group__MelSpecto
    group_MelS=[]
    for g in range(0, len(MelS_group_name_path)):
        group_MelS.append(ext.readKorean(MelS_group_name_path[g]))
        print('group_MelSpecto_PreProcessing done-', g)
    print('------Group_PreProcessing Finish!------')


#### target #####
    target_BW=[]
    target_Cen=[]
    target_MelS=[]
    print(ch_target_fileList_name)
    # target갯수 다른지 확인할 것
    for t in range(0,len(ch_target_fileList_name)):
        target_name.append(ch_target_fileList_name[t][:-18])
    # chroma
        target_c = ext.readKorean(ch_target_name_path[t])
        target_chroma_crop_img.append(ext.chroma_crop(target_c))
        print('target_C_cropping done-',t)
    # tonnetz
        target_t = ext.readKorean(to_target_name_path[t])
        target_tonnetz_crop_img.append(ext.tonnetz_crop(target_t))
        print('target_T_cropping done-',t)
    # sBW
        target_BW.append(ext.readKorean(BW_target_name_path[t]))
        print('target_sBW_preprocessing done-',t)
    # sCentroid
        target_Cen.append(ext.readKorean(Cen_target_name_path[t]))
        print('target_sCentroid_preprocessing done-',t)
    # sContra
        target_Ctra = ext.readKorean(Ctra_target_name_path[t])
        target_Ctra_crop_img.append(ext.sContra_crop(target_Ctra))
        print('target_sContra_cropping done-', t)
    # MelSpecto
        target_MelS.append(ext.readKorean(MelS_target_name_path[t]))
        print('target_MelSpecto_preprocessing done-', t)
    print('------Target_PreProcessing finish!-------')

    # -----------------------------------------------------------------------------------------------------------------

    print('ch_len_group:', len(group_chroma_crop_img), '// ch_len_target:', len(target_chroma_crop_img),
          '// chroma group:', int(len(group_chroma_crop_img)/group_size),'개')
    print('to_len_group:', len(group_tonnetz_crop_img), '// to_len_target:', len(target_tonnetz_crop_img),
            '// tonnetz group:', int(len(group_tonnetz_crop_img) / group_size), '개')
    print('sBW_len_group:', len(group_BW), '// sBW_len_target:', len(target_BW),
            '// spectral-BandWidth Group:', int(len(BW_group_name_path) / group_size), '개')
    print('sCentroid_len_group:', len(group_Centroid), '// sCentroid_len_target:', len(target_Cen),
          '// spectral-Centroid Group:', int(len(Cen_group_name_path) / group_size), '개')
    print('sContrast_len_group:', len(group_Ctra_crop_img), '// sContrast_len_target:', len(target_Ctra_crop_img),
          '// spectral-Contrast Group:', int(len(Ctra_group_name_path) / group_size), '개')
    print('MelSpecto_len_group:', len(group_MelS), '// MelSpecto_len_target:', len(target_MelS),
          '// MelSpecto Group:', int(len(group_MelS) / group_size), '개')

# Chromagram
    a = ext.ChromaHist(group_chroma_crop_img, target_chroma_crop_img, group_size, target_name, 'Chroma')
# Tonnetz
    b = ext.tonnetz_Contra_Hist(group_tonnetz_crop_img, target_tonnetz_crop_img, group_size, target_name, 'Tonnetz')
# sBW
    c = ext.histo_compare(group_BW, target_BW,group_size, target_name, 'BW')
# sCentroid
    d = ext.histo_compare(group_Centroid, target_Cen, group_size, target_name, 'Centroid')
# sContra
    e = ext.tonnetz_Contra_Hist(group_Ctra_crop_img, target_Ctra_crop_img, group_size, target_name, 'Contra')
# MelSpecto
    f = ext.histo_compare(group_MelS, target_MelS, group_size, target_name, 'MelSpecto')
    print('============Individual Excel Processing Finishing!============')

    total_Feature = pd.concat([a, b, c, d, e, f], axis=1, sort=True)
    total_Feature.columns = ['Chroma', 'Tonnetz', 'sBW', 'sCentroid', 'sContra', 'MelSpecto']
    total_Feature.index.names = ['제목']
    total_Feature.to_excel('C:/Users/s/Desktop/03_05_TOTAL_PYO.xlsx')

    print('============Total Excel Processing Finishing!============')