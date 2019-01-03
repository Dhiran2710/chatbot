import os
f = open('keywords.txt', 'w')
rootdir = './Product Images'
labelStr = ''
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        imgPath = os.path.join(subdir, file)
        path = os.path.dirname(imgPath)
        labelStr = os.path.basename(path)
        print('labelStr', labelStr)
        f.write(labelStr+'\n')
        break
f.close()
