# -*- coding: utf-8 -*-

#from nltk.tokenize import word_tokenize
#from nltk.tokenize import WordPunctTokenizer
#from tensorflow.keras import text_to_word_sequence
import kss
import pandas as pd
import os
from tkinter import filedialog

def fileOpen():
    fileName = filedialog.askopenfilename(initialdir = "/", title = "Select file", filetypes = (("*.xlsx","*xlsx"), ("*.xls","*xls"), ("*.csv","*csv"), ("all files", "*.*")))
    df = pd.read_excel(fileName)
    cat = list(df.columns)
    
    data = []
    for i in cat:
        data.append(list(df[i]))

    a = kss.split_sentences("".join(data[1][:100]))

    print(a)
    pass

def makeToken():
    pass



fileOpen()