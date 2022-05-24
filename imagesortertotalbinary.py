import os
import random

topfolder = 'images/'
infotext = 'annotations/list.txt'

file1 = open(infotext, 'r')
Lines = file1.readlines()


for idx,l in enumerate(Lines):
    s = l.split()
    if s[2] == str(2):
        species = 'dog/'
    else:
        species = 'cat/'
    r = random.random()
    if  r > 0.9:
        os.rename(topfolder + s[0] + '.jpg', 'sortedimages/test/' + species + s[0] + '.jpg')
    elif r > 0.85:
        os.rename(topfolder + s[0] + '.jpg', 'sortedimages/val/' + species + s[0] + '.jpg')
    else:
        os.rename(topfolder + s[0] + '.jpg', 'sortedimages/train/' + species + s[0] + '.jpg')
