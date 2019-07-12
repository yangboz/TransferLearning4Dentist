from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image

import argparse
import glob
import os
# https://towardsdatascience.com/learn-enough-python-to-be-useful-argparse-e482e1764e05
parser = argparse.ArgumentParser(description='normalized to den images')
#parser.add_argument('inPic', type=str, help='Input picture as den image gen template')
parser.add_argument('inDir', type=str, help='Input picture dir as den image gen template')
parser.add_argument('outDir', type=str, help='Output dir for den images normalized')
parser.add_argument('format', type=str, default='jpg', help='image format for den images normalized')
parser.add_argument('norW', type=int, default=200, help='normalized width')
parser.add_argument('norH', type=int, default=200, help='normalized height')

args = parser.parse_args()
#
print(args.inDir,args.outDir,args.format,args.norW,args.norH)


imgs = glob.glob(args.inDir+"*."+args.format)
print("imgs:",imgs)
#exit(0)
for img_ in imgs:
    print("gen...img_:",img_)
# img = load_img('data_dentisy/train/missing/IMG_6204.jpg')  # this is a PIL image
    #img = load_img(args.inPic)  # this is a PIL image
    img = load_img(img_)  # this is a PIL image
    ##
    im_resize = img.resize((args.norW,args.norH),Image.ANTIALIAS)
    print("resized image.size:",im_resize.size)
    ##
    im_resize.save(args.outDir+os.path.basename(img_) +"_"+ str(args.norW)+ "x"+str(args.norH)+"."+args.format, "JPEG")