import PySimpleGUI as sg
from rank_bm25 import *
import os


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import numpy as np
import pandas as pd

from collections import Counter
nltk.download('stopwords')
nltk.download('punkt')
sg.ChangeLookAndFeel('GreenMono')


# Global Functions
def remove_header(data):
    try:
        ind = data.index('\n\n')
        data = data[ind:]
    except:
        print("No Header")
    return data


def convert_lower_case(data):
    return np.char.lower(data)


def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words:
            new_text = new_text + " " + w
    return np.char.strip(new_text)


def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data


def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def remove_single_characters(data):
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if len(w) > 1:
            new_text = new_text + " " + w
    return np.char.strip(new_text)


def convert_numbers(data):
    data = np.char.replace(data, "0", " zero ")
    data = np.char.replace(data, "1", " one ")
    data = np.char.replace(data, "2", " two ")
    data = np.char.replace(data, "3", " three ")
    data = np.char.replace(data, "4", " four ")
    data = np.char.replace(data, "5", " five ")
    data = np.char.replace(data, "6", " six ")
    data = np.char.replace(data, "7", " seven ")
    data = np.char.replace(data, "8", " eight ")
    data = np.char.replace(data, "9", " nine ")
    return data


def stemming(data):
    stemmer = PorterStemmer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return np.char.strip(new_text)


def preprocess(data, query):
    if not query:
        data = remove_header(data)
    data = convert_lower_case(data)
    data = convert_numbers(data)
    data = remove_punctuation(data)  # remove comma seperately
    data = remove_apostrophe(data)
    data = remove_single_characters(data)
    data = stemming(data)
    return data


def Tokenizing(data):
    processed_text = []
    for doc in data:
        processed_text.append(word_tokenize(str(preprocess(doc, True))))
    return processed_text


class GUI:
    def __init__(self, Queries):
        self.layout: list = [
            [sg.Text('Search Query', size=(11, 1)),
             sg.Input('', size=(40, 1), focus=True, key="QUERY"),
             sg.Radio('Results=5', size=(10, 1),
                      group_id='choice', key="_5", default=True),
             sg.Radio('Results=10', size=(10, 1),
                      group_id='choice', key="_10"),
             sg.Radio('Results=20', size=(10, 1), group_id='choice', key="_20")],
            [sg.Combo(Queries, size=(55, 1), key="QUERY_DROP"),
                # sg.Input('/..', size=(40,1), key="PATH"),
                # sg.FolderBrowse('Browse', size=(10,1)),
                # sg.Button('Re-Index', size=(10,1), key="_INDEX_"),
                sg.Button('Search', size=(10, 1), bind_return_key=True, key="_SEARCH_")],
            [sg.Output(size=(100, 30))]]

        self.window: object = sg.Window(
            'FORNAX Search Engine', self.layout, element_justification='left')


class SearchEngine:
    def __init__(self):
        self.file_index = []
        self.results = []
        self.matches = 0
        self.records = 0

    def Input_processing(self):
        with open('./Input/CISI.ALL') as f:
            lines = ""
            for l in f.readlines():
                lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
            lines = lines.lstrip("\n").split("\n")

        doc_set = {}
        doc_id = ""
        doc_text = ""
        doc_set[0] = ""
        for l in lines:
            if l.startswith(".I"):
                doc_id = l.split(" ")[1].strip()
            elif l.startswith(".X"):
                doc_id = int(doc_id)
                doc_set[doc_id] = doc_text.lstrip(" ")
                doc_id = ""
                doc_text = ""
            else:
                # The first 3 characters of a line can be ignored.
                doc_text += l.strip()[3:] + " "

        docs = (list(doc_set.values()))

        return (docs)

    def Query_File_processing(self):
        with open('./Input/CISI.QRY') as f:
            lines = ""
            for l in f.readlines():
                lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
            lines = lines.lstrip("\n").split("\n")

        qry_set = {}
        qry_id = ""
        qry_set[0] = ""
        for l in lines:
            if l.startswith(".I"):
                qry_id = l.split(" ")[1].strip()
            elif l.startswith(".W"):
                qry_id = int(qry_id)
                qry_set[qry_id] = l.strip()[3:]
                qry_id = ""

        queries = (list(qry_set.values()))
        return (queries)

    def Relation_Set(self):
        rel_set = {}
        qry_id = ""
        rel_set[0] = ""
        with open('./Input/CISI.REL') as f:
            for l in f.readlines():
                qry_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0]
                doc_id = l.lstrip(" ").strip(
                    "\n").split("\t")[0].split(" ")[-1]
                qry_id = int(qry_id)
                if qry_id in rel_set:
                    rel_set[qry_id].append(doc_id)
                else:
                    rel_set[qry_id] = []
                    rel_set[qry_id].append(doc_id)

        relations = (list(rel_set.values()))
        return (relations)

    def BM_25_processor(self, k, query, Docs):
        preprocessed_query = preprocess(query, True)
        tokenized_query = word_tokenize(str(preprocessed_query))

        processed_text = []

        for doc in Docs:
            processed_text.append(word_tokenize(str(preprocess(doc, True))))
        bm25 = BM25Okapi(processed_text)
        rank_doc = bm25.get_top_n(tokenized_query, Docs, n=k)

        index = []
        for doc in rank_doc:
            i = Docs.index(doc)
            index.append(i)
        return index


def main():
    s = SearchEngine()

    k = 0
    Docs = s.Input_processing()
    Queries = s.Query_File_processing()
    Relations = s.Relation_Set()

    g = GUI(Queries)

    while True:
        event, values = g.window.read()
        if(values['_5']):
            k = 5
        elif(values['_10']):
            k = 10
        elif(values['_20']):
            k = 20

        if event is None:
            break

        if event == 'QUERY_DROP':
            values['QUERY'] = values['QUERY_DROP']

        if event == '_SEARCH_':
            query = values['QUERY']
            l = s.BM_25_processor(k, query, Docs)
            for i in l:
                print("Document:", i)
                print(Docs[i])
                print(
                    '---------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
