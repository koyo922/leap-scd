import re
import numpy as np
import gen_notebook as gen

f=open('/home/sriram/speech/OverLap/TIMIT/lists/validation_files.list')
f=f.read()
f=f.strip()
f=re.split('\n',f)

for i in range(10000):
        [index1,index2]=np.random.choice(496,size=2,replace=False)
        file1=f[index1]
        file2=f[index2]
        gen.gen_func(file1,file2,i)
        print "Completed Generating",  i+1
