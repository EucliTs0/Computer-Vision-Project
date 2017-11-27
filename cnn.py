from os import listdir
from PIL import Image
import xml.etree.ElementTree as ET
from keras.applications.vgg19 import VGG19
from numpy import asarray, array, ravel

image_data = "../../MATLAB/cv_project_sem2/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/"
image_annotations = "../../MATLAB/cv_project_sem2/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/"

def load_images(img_folder):
    images = {}
    i = 0
    for image in listdir(img_folder):
        img_num = image[:-4]
        images[img_num] = [asarray(Image.open(img_folder + image).resize((256,256))) / 255.0]
        i += 1
        if i % 500 == 0:
            print(i)
    return images

def load_labels(label_folder):
    labels = {}
    for image in listdir(label_folder):
        img_num = image[:-4]
        tree = ET.parse(label_folder + img_num + ".xml")
        root = tree.getroot()
        objects = [x.text for x in root.findall("./object/name")]
        labels[img_num] = set(objects)

def get_features(imgs, model):
    i = 0
    features = []
    for image in imgs:
        i += 1
        features.append(ravel(model.predict(asarray(imgs[image]))))
        if i % 500 == 0:
            print(i)
    keys = list(imgs.keys())
    return keys, features

def write_features(keys, features, filename):
    f = open(filename, 'w')
    n = len(keys)
    for i in range(n):
        f.write(keys[i])
        for num in features[i]:
            f.write(" " + str(num))
        f.write("\n")
    f.close()

imgs = load_images(image_data)
print(len(imgs))
mod = VGG19(include_top=False,input_shape=(256,256,3),pooling='max')
keys, features = get_features(imgs, mod)
print(len(imgs))
write_features(keys, features, "normalized_features.txt")