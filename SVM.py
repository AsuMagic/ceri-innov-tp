# from audioop import lin2adpcm
from collections import defaultdict
# import csv
# import sys
# from glob import glob
# from operator import delitem
# from time import sleep
from tkinter import W
# from unittest import result
import pandas as pd
import re
import os
import unicodedata
from tqdm import tqdm

tqdm.pandas()

from gensim.test.utils import common_texts
from gensim.models import Word2Vec

# reciew_id movie name user_id note commentaire

noteSomme = defaultdict(lambda:0)

# read the data from bin.zst thanks to asu 
def readXml(name):
    data = pd.read_pickle(name)
    
    # print(data)

    return data


# extract all word to a biblio in form of index => word{nb_appaer,id}
def extracLexique(data):
    id=1
    lexique = defaultdict(lambda:[0,0])

    for index, row in data.iterrows():
        tmp = treatRegex(row["commentaire"])
        for tmp2 in tmp:
            lexique[tmp2][0]+=1
            lexique[tmp2][1]=id
            id+=1
    return lexique # pd.DataFrame(lexique)

# we create the file SVM wich name is typeData
def converter(typeData, data, lexique,lexiconSentiment):
    SVM=[]

    for index, row in data.iterrows():
        list=[]
        list_wordID=[]
        ligne=str(getOpinionValue(row["note"]))+" "
        noteSomme[getOpinionValue(row["note"])]+=1
        words = treatRegex(row["commentaire"])

        for word in words:
            if not lexique.get(word)[1] in list_wordID:
                list.append((lexique.get(word)[1],lexique.get(word)[0]))
                list_wordID.append(lexique.get(word)[1])

        list=sorted(list,key=lambda list: list[0])
        for item in  list:
            ligne+= str(item[0])+":"+str(item[1])+" "
        SVM.append(ligne)

    file = open(typeData+".svm", "w")
    for ligne in SVM:
        file.write(ligne)
        file.write("\n")
    file.close()

# use in the converte function to find the class of a commentary
def getOpinionValue(x):
    # return 2.5 3.5

    # if x < 2:  # 0.5 1 1.5
    #     return 1
    # elif x==2 : # 2 
    #     return 2
    # elif x> 2: # 2.5 3 3.5
    #     return 3
    if x == 0.5:
        return 1
    elif x == 1:
        return 2
    elif x == 1.5:
        return 3
    elif x == 2:
        return 4
    elif x == 2.5:
        return 5
    elif x == 3:
        return 6
    elif x == 3.5:
        return 7
    elif x == 4:
        return 8
    elif x == 4.5:
        return 9
    elif x == 5:
        return 10

    
    # if x < 2:  # 0.5 1 1.5
    #     return 1
    # elif x>=2 and x<=4: # 2 2.5 3 3.5 
    #     return 2
    # elif x> 4: # 4 4.5 5
    #     return 3
    # print(x)
    return x

# treat a commentary to get all word and apply treatment to them
def treatRegex(txt):
    # txt = re.sub("([-])",' ',txt)
    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    # txt = re.sub("([\"<>@<>().',#@\[\]/])",'', txt) #47.49 
    tmp = re.findall("(\w[\w']*)", txt)# get all word
    
    for index,word in enumerate(tmp):# lower them and remove accent
        tmp[index]= word.lower()
        tmp[index]= strip_accents(word)

    # #use with bash to research symbole in string
    # tmp += re.findall(sys.argv[1],txt)

    # # normaly add accuracy (symbole important)
    # tmp+= re.findall("([\"<>#])",txt)
    # print(tmp)
    return tmp

# bas:51.67 

# use to remove accent from a word get fucntino from (maiby rework it)
# https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

# get a list a stopword and remove them use in treatregex
# to rework if need
def removeStopWord(data,colunm): # long a conter

    stopWord =[]
    with open("stopWord","r") as file:
        stopWord = [line.rstrip() for line in file]

    for index, row in data.iterrows():
        tmp = treatRegex(row[colunm])
        result=""
        for word in tmp:
            # word = word.lower()
            if not word in stopWord:
                result+=word+" "
        # print(tweets)

        data.loc[index, [colunm]] = [result]

    # print(data)
    return data

# get a lexiqueSentiment of good and bad word from 2 file
def MakerLexiconSentiments():
    
    Column=['word'] 

    folder = "Lexicon-Sentiment/"
    read = pd.read_csv(folder+"positive_words_fr.txt", names=Column, header=None)
    # read =removeStopWord(read,"word")
    read['sentiment']='positive'
    #read["tweets"] = read["tweets"].str.lower()
    # read.to_csv('Lexicon.csv',index=False, sep='\t')

    read2 = pd.read_csv(folder+"negative_words_fr.txt", names=Column, header=None)
    # read =removeStopWord(read,"word")
    read2['sentiment']='negative'
    #read["tweets"] = read["tweets"].str.lower()
    
    read = pd.concat([read,read2])
    read.to_csv('Lexicon.csv',index=False, sep='\t')
    # print(read)
    read = read.set_index('word')
    return read

    # return NULL





def MyWord2Vec(data):

    listWord=[]
    print(common_texts)
    # window of 4 so 2 before and 2 after 
    
    for index, row in data.iterrows():
        # words = treatRegex(row["commentaire"])
        txt = row["commentaire"]
        # txt = re.sub("([\"<>@<>().',#@\[\]/])",'', txt) #47.49 
        tmp = re.findall("(\w[\w']*)", txt)# get all word
        
        # print(tmp)
        for index,word in enumerate(tmp):# lower them and remove accent
            
            tmp[index]= word.lower()
            tmp[index]= strip_accents(word)
            # si debut
            listWord.append(tmp[max(0,index-2):index+3])
            # print(listWord)
            

        # print("END")
        # sleep(3)
        # print(tmp)
        # listWord+=tmp

    print(len(listWord)) # pourquoi par retouné la liste de mot afin de ne plus traité sa 


    # print(common_texts)
    # model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
    model = Word2Vec(sentences=listWord, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")

    # model = Word2Vec.load("word2vec.model")
    model.train([["hello", "world"]], total_examples=1, epochs=1)

    print(model.wv["Hoffman"])
    print(model)


if __name__ == "__main__":

    lexiconSentiment=""
    # lexiconSentiment = MakerLexiconSentiments()
    
    # name of to train and test/dev 
    nametrain = "train.bin.zst"
    namedev = "dev.bin.zst"

    if os.path.exists("train.svm"):
        os.remove("train.svm")
    if os.path.exists("dev.svm"):
        os.remove("dev.svm")


    # make the train.svm
    data = readXml(nametrain)
    data = data.reset_index()
    # MyWord2Vec(data)
    lexique = extracLexique(data)
    converter("train",data,lexique,lexiconSentiment)

    # make the dev.svm
    data = readXml(namedev)
    data = data.reset_index()
    lexique = extracLexique(data)
    converter("dev",data,lexique,lexiconSentiment)

# print(noteSomme)
# remove accent ls

# rajouter un lexique de sentiment de mot 
# racourcie les abuue de lette ex: looool -> lool 

#Accuracy = 57.7406% (552/956)

# Accuracy = 77.4691% (77779/100400)


# ecrate type des film 
# mot souvent utilsier dans une certain class
# nuage de tag couleur lier au mot positif,negatif
# graphe du nombre de review