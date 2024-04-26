import os
import shutil
def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            if(fullPath.endswith('.webp')):
                allFiles.append(fullPath)
                
    return allFiles

dirname = 'C:\Codes\Codes\Python\Random progs\convertor\two-way-sign-language-translator' #Your Full Path of Projects Folder
dest="C:\Codes\Codes\Python\Random progs\convertor\two-way-sign-language-translator/filtered_data"

data=getListOfFiles(dirname)
for i in range(len(data)):
    fname=dest+str(i)+".webp"
    shutil.copyfile(data[i], fname)
