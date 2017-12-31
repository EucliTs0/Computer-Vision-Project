from os import listdir
import xml.etree.ElementTree as ET
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import average_precision_score
from numpy import mean, zeros


def load_labels(label_folder):
    labels = {}
    classes = set()
    for image in listdir(label_folder):
        img_num = image[:-4]
        tree = ET.parse(label_folder + img_num + ".xml")
        root = tree.getroot()
        objects = [x.text for x in root.findall("./object/name")]
        for o in objects:
            classes.add(o)
        labels[img_num] = set(objects)
    return labels, classes

def load_features(filename):
    f = open(filename, 'r')
    feats = {}
    for line in f:
        features = line.split()
        feats[features[0]] = [float(x) for x in features[1:]]
    f.close()
    return feats

def make_multi_label(features, labels, objects):
    objects = list(objects)
    objects.sort()
    X = []
    Y = []
    for image in labels:
        y = zeros(20)
        for ob in labels[image]:
            y[objects.index(ob)] = 1.0
        Y.append(y)
        X.append(features[image])
    return X, Y

# function that takes dictionary of (imageID, features) and (imageID, labels)
# returns dataset with binary labels based on if "classname" object is in the image
def make_binary_dataset(features, labels, classname):
    X = []
    Y = []
    for image in labels:
        X.append(features[image])
        if classname in labels[image]:
            Y.append(1.0)
        else:
            Y.append(0.0)
    return X, Y

def normalize(features):
    keys = list(features.keys())
    feats = list(features.values())
    scaler = StandardScaler(copy=False)
    feats = scaler.fit_transform(feats)
    features = {}
    n = len(keys)
    for i in range(n):
        features[keys[i]] = feats[i]
    return features



if __name__ == '__main__':
    image_annotations = "../../MATLAB/cv_project_sem2/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/"
    features = load_features("train_features.txt")
    labels, classes = load_labels(image_annotations)
    features = normalize(features)

    avg_prec = []
    i = 0
    for object in classes:
        i += 1
        print(object + " " + str(i))
        X, Y = make_binary_dataset(features, labels, object)
        svc = SVC(class_weight='balanced', probability=True)
        parameters = {'kernel': ['rbf'], 'C': [0.5,1.5,2.5,3.5], 'gamma': [0.001]}
        clf = GridSearchCV(svc, parameters, scoring='average_precision', verbose=1, n_jobs=4, cv=2)
        clf.fit(X,Y)
        print(clf.best_params_)
        print(clf.best_score_)
        avg_prec.append(clf.best_score_)

        #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
        #clf = MLPClassifier(solver='adam', hidden_layer_sizes=(3,2,), activation='relu')
        # clf.fit(X_train, y_train)
        # pred = clf.predict_proba(X_train)
        # print(average_precision_score(y_train, [x[1] for x in pred]))
        # pred = clf.predict_proba(X_test)
        # ap = average_precision_score(y_test, [x[1] for x in pred])
        # print(ap)
        # avg_prec.append(ap)

    print(mean(avg_prec))


    # X, Y = make_multi_label(features, labels, classes)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    # clf = MLPClassifier(solver='adam', hidden_layer_sizes=(30,), activation='relu')
    # print("training")
    # clf.fit(X_train, y_train)
    # pred = clf.predict_proba(X_train)
    # print(pred[0])
    # print(average_precision_score(y_train, pred))
    # pred = clf.predict_proba(X_test)
    # ap = average_precision_score(y_test, pred)
    # print(ap)
    #
# exit(0)