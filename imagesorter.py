import os

topfolder = 'images/'
infotext = 'annotations/trainval.txt'

file1 = open(infotext, 'r')
Lines = file1.readlines()
split = len(Lines)
i = 0
print(split)

for l in Lines:
    s = l.split()
    if s[2] == str(2):
        species = 'dog/'
    else:
        species = 'cat/'
    if i < 2700:
        os.rename(topfolder + s[0] + '.jpg', 'sortedimages/train/' + species + s[0] + '.jpg')
    else:
        os.rename(topfolder + s[0] + '.jpg', 'sortedimages/val/' + species + s[0] + '.jpg')
    i =i+1