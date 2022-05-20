import os
import random

topfolder = 'images/'
infotext = 'annotations/list.txt'

file1 = open(infotext, 'r')
Lines = file1.readlines()


for i in range(37):
    newtrain ='sortedimages/train/' + str(i+1)
    newval = 'sortedimages/val/' + str(i+1)
    newtest = 'sortedimages/test/' + str(i+1)
    if not os.path.exists(newtrain):
        os.makedirs(newtrain)
    if not os.path.exists(newval):
        os.makedirs(newval)
    if not os.path.exists(newtest):
        os.makedirs(newtest)

for idx,l in enumerate(Lines):
    s = l.split()
    breed = str(s[1])+"/"
    r = random.random()
    if  r > 0.9:
        os.rename(topfolder + s[0] + '.jpg', 'sortedimages/test/' + breed + s[0] + '.jpg')
    elif r > 0.85:
        os.rename(topfolder + s[0] + '.jpg', 'sortedimages/val/' + breed + s[0] + '.jpg')
    else:
        os.rename(topfolder + s[0] + '.jpg', 'sortedimages/train/' + breed + s[0] + '.jpg')
