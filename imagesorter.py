import os

topfolder = 'images/'
infotext = 'annotations/trainval.txt'

file1 = open(infotext, 'r')
Lines = file1.readlines()
split = len(Lines)
print(split)

for i in range(37):
    newtrain ='sortedimages/train/' + str(i+1)
    newval = 'sortedimages/val/' + str(i+1)
    if not os.path.exists(newtrain):
        os.makedirs(newtrain)
    if not os.path.exists(newval):
        os.makedirs(newval)

for idx,l in enumerate(Lines):
    s = l.split()
    breed = str(s[1])+"/"
    if idx%11 == 0:
        os.rename(topfolder + s[0] + '.jpg', 'sortedimages/val/' + breed + s[0] + '.jpg')
    else:
        os.rename(topfolder + s[0] + '.jpg', 'sortedimages/train/' + breed + s[0] + '.jpg')