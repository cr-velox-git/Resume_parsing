# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 09:12:18 2022

@author: cr_velox
"""
import pickle
import PyPDF2  

model = pickle.load(open(r"resume_model_1","rb"))
#use pip install PyMuPDF
#PyPdf2 doesnot work it extraced text, but not all  info and sending garbed value some time

pdf_location = r"C:\Users\cr_velox\Desktop\RESUME_of_Rajesh_khan.pdf"

# creating a pdf file object 
pdfFileObj = open(pdf_location, 'rb') 
    
# creating a pdf reader object 


reader = PyPDF2.PdfFileReader(pdfFileObj)
pageObj = reader.getNumPages()
page_data=''
for page_count in range(pageObj):
    page = reader.getPage(page_count)
    page_data = page_data + page.extractText()

    
tx = " ".join(page_data.split('\n'))   
doc = model(tx)

for ent in doc.ents:
    print(ent)
    print({ent.label_.upper():{30}}- {ent.page_data})
    
 