# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:04:37 2019

@author: Eric Bianchi
"""
import os
from shutil import copy
from doc2docx import doc2docx

# This is for converting doc files to docx files. 
def sort_files(directory, extension):
    """
    Parameters
    ----------
    [directory] : string
        path to directory to be sorted over.
    extension : string
        extension that is to be sorted out (.doc)

    Returns
    -------
    [None.]

    """
    for dirpath, dirnames, files in os.walk(directory):
        for name in files:
            if name.startswith("~"):
                continue
            elif extension and name.endswith(extension):
                #print(os.path.join(dirpath, name))              
                path = dirpath + "/" + name
            elif not extension:
                #print(os.path.join(dirpath, name))
                print("not a: " + extension)
                



