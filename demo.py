from sklearn.externals import joblib
from os import listdir
from PIL import Image
from keras.applications.vgg19 import VGG19
from numpy import asarray, ravel, mean, zeros

def normalize(features):
    scaler = joblib.load('models/feature_scalar.pkl')
    features = scaler.transform(features)
    return features

def detect_objects(image_file_name):
    ## extract and normalize features from pre-trained model
    model = VGG19(include_top=False, input_shape=(224, 224, 3), pooling='max')
    im = asarray(Image.open(image_file_name).resize((224, 224)))
    new_img = zeros((224, 224, 3))
    for i in range(3):
        new_img[:, :, i] = im[:, :, i] - mean(im[:, :, i])
    del im
    features = normalize([ravel(model.predict(asarray([new_img])))])[0]
    predictions = []
    ## test features on each object's trained SVMs
    for model_file in listdir('models/'):
        if model_file[-7:] == 'svm.pkl':
            object = model_file[:-7]
            svm = joblib.load('models/' + model_file)
            pred = svm.predict_proba([features])[:,1]
            predictions.append((object, pred[0]))
    return predictions



pred = detect_objects('bottle.jpg')
for tup in pred:
    if tup[1] > 0.6:
        print(tup[0])