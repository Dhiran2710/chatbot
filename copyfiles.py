import os
import shutil

rootdir = './Product Images'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        imgPath = os.path.join(subdir, file)
        path = os.path.dirname(imgPath)
        labelStr = os.path.basename(path)
        dest = os.path.join('downloads', labelStr)
        if(os.path.isfile(imgPath)):
            shutil.copy(imgPath, dest)
        
        
