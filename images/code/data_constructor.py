import os
import gloffice as gl
Dict = gl.partDict;
def ListFilesToTxt(dir_,trainfile,testfile,wildcard,recursion,sdomain,tdomain):
    exts = wildcard.split(" ")
    files = os.listdir(dir_)
    files.sort();
    for name in files:
        fullname=os.path.join(dir_,name)
        if(os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname,trainfile,testfile,wildcard,recursion,sdomain,tdomain)
        else:
            for ext in exts:
                if(name.endswith(ext)):
                    #amazon to caltech
                    dirsplit = dir_.split('/')
                    #domain = dirsplit[-3]
                    domain = dirsplit[-2]
                    #datatype = dirsplit[-2]
                    classes = dirsplit[-1]
                    #if datatype == "train":
                    if classes in Dict:
                        if (domain == sdomain):
                            trainfile.write(fullname + " "+"-1"+" "+Dict[dirsplit[-1]]+"\n")
                        elif(domain == tdomain):
                            trainfile.write(fullname+" "+Dict[dirsplit[-1]]+" "+"-1"+"\n")
                            testfile.write(fullname+ " "+Dict[dirsplit[-1]]+ " "+ "-1"+"\n")
                        else:
                            pass
                    break

os.chdir("/home/hsi/mmd-caffe/images")#change to the directory of images
dir_= os.getcwd();#get the current work
#print "input source domain:"
sdomain = input('input source domain:')
#print "input target domain:"
tdomain = input('input target domain:')
filesuffix = sdomain[0]+str(2)+tdomain[0]+'.txt'
trainfile='train'+filesuffix
testfile = "test"+filesuffix
wildcard = ".bmp .jpg"
trainfile = open(trainfile,"w")
testfile = open(testfile,'w')
if not trainfile:
    print ("cannot open the file %s for writing" % outfile)
ListFilesToTxt(dir_,trainfile,testfile,wildcard, 1,sdomain,tdomain)
trainfile.close()
testfile.close()
