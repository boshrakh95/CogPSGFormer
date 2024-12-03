import glob2
import matplotlib.pyplot as plt
import pydicom
import os
import gc

files = '/media/livia/My Passport/CT XRAY/DICOM/PA000001/ST000004/SE000002'
dir1 = sorted(glob2.glob(files + '/IM*'))

dir2 = '/home/livia/Desktop/images/2'
os.mkdir(dir2)
for i in range(len(dir1)):
    dataset = pydicom.dcmread(dir1[i])
    plt.imshow(dataset.pixel_array, cmap='gray')
    plt.savefig(dir2 + '/image' + str(i) + '.png')
    plt.close()
    del dataset
    gc.collect()
