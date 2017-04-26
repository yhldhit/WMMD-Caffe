import os
import gloffice as gl

def ListFilesToTxt(dir,trainfile,testfile,wildcard,recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    files.sort();
    countDict = {};
    #targetDict = {};
    for name in files:
        fullname=os.path.join(dir,name)
        if(os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname,trainfile,testfile,wildcard,recursion)
        else:
            for ext in exts:
                if(name.endswith(ext)):
                    #amazon to caltech
                    dirsplit = dir.split('/')
                    domain = dirsplit[-2]
                    #datatype = dirsplit[-2]
                    classes = dirsplit[-1]
                    #if datatype == "train":
                    if classes in gl.Dict:
                        #if (domain == 'dslr'):
                        if classes in countDict:
                            countDict[classes] = countDict[classes] +1
                        else:
                            countDict[classes] = 1;
                                #trainfile.write(fullname + " "+"-1"+" "+gl.partDict[dirsplit[-1]]+"\n")
                            #elif(domain == 'caltech256'):
                             #   if classes in targetDict:
                              #      targetDict[classes] = targetDict[classes] +1;
                               # else:
                               #     targetDict[classes] = 1;
                                #trainfile.write(fullname+" "+gl.partDict[dirsplit[-1]]+" "+"-1"+"\n")
                                #testfile.write(fullname+ " "+gl.partDict[dirsplit[-1]]+ " "+ "-1"+"\n")
                            #else:
                             #   pass
                    break
    if (len(countDict)>0):
       # print"*********source************
        for name in countDict:
            print domain,name,countDict[name]

    #if (len(targetDict)>0):
    #    print"*********target*************"
     #   for name in targetDict:
      #      print name,targetDict[name]


dir = os.chdir('/home/yhl/mmd-caffe/images')
dir= os.getcwd();
trainfile="traincount.txt"
testfile = "testcount.txt"
wildcard = ".bmp .jpg"
trainfile = open(trainfile,"w")
testfile = open(testfile,'w')
if not trainfile:
    print ("cannot open the file %s for writing" % outfile)
ListFilesToTxt(dir,trainfile,testfile,wildcard, 1)
trainfile.close()
testfile.close()
