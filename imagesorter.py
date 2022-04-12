import os

topfolder = 'images/'
infotext = 'annotations/trainval.txt'

file1 = open(infotext, 'r')
Lines = file1.readlines()

for l in Lines:
    s = l.split()
    if s[2] == str(2):
        species = 'dog/'
    else:
        species = 'cat/'
    os.rename(topfolder + s[0] + '.jpg', 'sortedimages/' + species + s[0] + '.jpg')