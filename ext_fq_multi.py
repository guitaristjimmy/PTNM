from multiprocessing import Process
import multiprocessing
import os
import numpy as np
from extract_fq import ExtractFrequency
import pandas as pd
from pandas import DataFrame

folder_path = 'E:/topickle/송희바꿈'

Ext_Freq = ExtractFrequency()
Ext_Freq.folder_name_load(folder_path)
Ext_Freq.file_name_load(folder_path)

num_cpu = multiprocessing.cpu_count()


def action(i):
    tempo_info = Ext_Freq.load_and_extract_tempo(Ext_Freq.file_name_path[i])
    print("tempo extratction", i, "finished")
    melody_info = Ext_Freq.load_and_extract_melody(Ext_Freq.file_name_path[i])
    print("melody extratction", i, "finished")
    Ext_Freq.to_pickle(Ext_Freq.file_list_name[i], tempo_info, melody_info, i)



if __name__ == '__main__':

    file_count = len(Ext_Freq.file_name_path)

    jobs = []
    for i in range(0, file_count):
        process = Process(target=action, args=(i,))
        jobs.append(process)
        process.start()
        print('process' + str(i) + 'start')

    for proc in jobs:
        proc.join()
        print('process done')

