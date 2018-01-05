from os import listdir
from PIL import Image
import xml.etree.ElementTree as ET
from keras.applications.vgg19 import VGG19
from numpy import asarray, array, ravel, mean, zeros

image_data = "../../MATLAB/cv_project_sem2/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/"
image_annotations = "../../MATLAB/cv_project_sem2/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/"

def compute_save_features(img_folder, model, filename):
    f = open(filename, 'w')
    i = 0
    for image in listdir(img_folder):
        img_num = image[:-4]
        im = asarray(Image.open(img_folder + image).resize((224,224)))
        new_img = zeros((224,224,3))
        for i in range(3):
            new_img[:,:,i] = im[:,:,i] - mean(im[:,:,i])
        features = ravel(model.predict(asarray([new_img])))
        f.write(img_num)
        for num in features:
            f.write(" " + str(num))
        f.write("\n")
        i += 1
        if i % 100 == 0:
            print(i)
    f.close()

mod = VGG19(include_top=False,input_shape=(224,224,3),pooling='max')
compute_save_features(image_data, mod, "test_features.txt")
