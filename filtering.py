# -*- coding: cp949 -*-

from collections import Counter
import pandas as pd
import numpy as np
from tkinter import filedialog
import ast
import os
import re

def fileOpen():
    global fileName
    fileName = filedialog.askopenfilename(initialdir = "/", title = "Select file", filetypes = (("*.xlsx","*xlsx"), ("*.xls","*xls"), ("*.csv","*csv"), ("all files", "*.*")))

    df = pd.read_excel(fileName)
    dt = df.values.tolist()
    return dt

def fileCreate(dt, val):
    global fileName

    try:
        folder = '10000recipeData'  # ���� ���� ������
        os.mkdir(folder)
    except:
        pass

    pattern = r'F\d+_T\d+'
    fName = re.findall(pattern, fileName)[0]

    fName = f'{folder}/{fName}_{val}.xlsx'

    cols = [['Key', '�丮��', '�κ�', '�ҿ�ð�', '���̵�', '���', '������'], ['���', '�󵵼�']]
    df = pd.DataFrame(dt, columns = cols[val - 1])

    df.to_excel(fName, index = False)

def process():
    dt = fileOpen()

    subDt = []
    for i in dt:
        min = i[3].find('��')
        if min != -1 and int(i[3][ : min]) <= 30 and i[4] in ['�ƹ���', '�ʱ�']:
            subDt.append(i)

    fileCreate(subDt, 1)
    subDt = np.array(subDt)

    ingred = np.transpose(subDt[ : , 5 : 6]).tolist()[0]
    ingred_dict = []
    for i in ingred:
        temp = ast.literal_eval(i)
        ingred_dict.append(temp)

    ingreds = []
    for i in ingred_dict:
        for j in i:
            ingreds.append(list(j.keys())[0])

    counts = Counter(ingreds)

    counts_key = list(counts.keys())
    counts_val = list(counts.values())

    param = []
    for i in range(len(counts_key)):
        param.append([counts_key[i], counts_val[i]])  
    
    fileCreate(param, 2)

while 1:
    try:
        process()
    except:
        break