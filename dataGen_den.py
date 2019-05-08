from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import argparse
# https://towardsdatascience.com/learn-enough-python-to-be-useful-argparse-e482e1764e05
parser = argparse.ArgumentParser(description='dataGen to den images')
parser.add_argument('inPic', type=str, help='Input picture as den image gen template')
parser.add_argument('outDir', type=str, help='Output dir for den images gen')
parser.add_argument('prefix', type=str, default='den_', help='saved prefix for den image gen file name')
parser.add_argument('format', type=str, default='jpg', help='image format for den images gen')
parser.add_argument('gens', type=int, default=100, help='number of images to gen')

args = parser.parse_args()
print(args.inPic,args.outDir,args.prefix,args.format,args.gens)

# from keras.utils.training_utils import multi_gpu_model
# import tensorflow as tf
print("datagen init... ")
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# img = load_img('data_dentisy/train/missing/IMG_6204.jpg')  # this is a PIL image
img = load_img(args.inPic)  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

MAX_GEN = args.gens
print("datagen in MAX ",MAX_GEN)
# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=10,
                          # save_to_dir='data_dentisy/train/missing/gen', save_prefix='den', save_format='jpg'):
                            save_to_dir=args.outDir, save_prefix=args.prefix, save_format=args.format):
    print("gen-ed index:",i)
    i += 1
    if i > MAX_GEN:
        break  # otherwise the generator would loop indefinitely