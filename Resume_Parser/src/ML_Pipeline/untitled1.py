# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 13:59:42 2022

@author: cr_velox
"""

from asyncio.windows_utils import pipe
from doctest import DONT_ACCEPT_BLANKLINE
import logging
from pickletools import optimize
import spacy
import random
import pickle
import json
from tika import parser
import os
from spacy.tokens import Doc, DocBin
from spacy.cli.train import train
from spacy.training import Example
from spacy.lang.en import English

def convert_data_to_spacy(JSON_FilePath):
    try:
        training_data = []
        lines=[]
        with open(JSON_FilePath, 'r',encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                #only a single point in text annotation.
                point = annotation['points'][0]
                labels = annotation['label']
                # handle both list of labels or a single label.
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    
                    entities.append((point['start'], point['end'] + 1 ,label))


            training_data.append((text, {"entities" : entities}))
        print('train data converted from json')
        return training_data
    except Exception as e:
        logging.exception("Unable to process " + JSON_FilePath + "\n" + "error = " + str(e))
        return None



def model(TRAIN_DATA):
    nlp = English()
    
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner', last=True)
    else:
        ner = nlp.get_pipe('ner')
        
        
    #add label
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
            
    
    optimizer = nlp.initialize()
    for itn in range(1):
        print("startting iteration: "+str(itn))
        losses = []
        random.shuffle(TRAIN_DATA)
        for text, annotations in TRAIN_DATA:
            try:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.2, sgd=optimizer, losses=losses)
    
            except Exception as e:
                print("exception: "+str(itn)+"/n"+str(e)+"/n"+str(annotations))
                pass
        print(losses)
        
    nlp.to_disk("/output")
    return nlp

train_data= convert_data_to_spacy("E:\\Python Project\\resume_parcer\\Resume_Parser\\input\\training\\Entity Recognition in Resumes.json")
#model = pickle.load(open(r"resume_model","rb"))
resume_model = model(train_data)


