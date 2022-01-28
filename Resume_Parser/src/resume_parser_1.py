# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 07:09:21 2022

@author: cr_velox
"""

import spacy
import pickle #file formate 
import random
from ML_Pipeline import json_spacy



#training data
train_data = json_spacy.convert_data_to_spacy("E:\\Python Project\\resume_parcer\\Resume_Parser\\input\\training\\Entity Recognition in Resumes.json")

nlp = spacy.blank('en')

#model
def train_model(train_data):
    #entity recognisation model
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last = True)
       # ruler = nlp.add_pipe("entity_ruler")
        
    #adding label in the pipe line
    for _, annotation in train_data:
        for ent in annotation['entities']:
            ner.add_label(ent[2])
            
    #check for other pipe line
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes): #only train NER
        optimizer = nlp.begin_training()
        for itn in range(100):
            print("Starting iteration" + str(itn))
            random.shuffle(train_data)
            losses = {}
            index = 0
            for text, annotations in train_data:
                #print(index)
                try:
                    nlp.update(
                        [text], #batch of text
                        [annotations], #batch of annotation
                        drop = 0.2, # dropout - make it harder to memorise 
                        sgd = optimizer, #callable to update the weight
                        losses=losses)
                except Exception as e:
                    pass
            #print(text)
            #print(annotation)
                        
            print(losses)
            
train_model(train_data)

#save the model
nlp.to_disk('nlp_model')
#print("save model")

pickle.dump(nlp, open(r"resume_model_1","wb"))
#load an train model+