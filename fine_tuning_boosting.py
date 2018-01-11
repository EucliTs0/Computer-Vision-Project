from os import listdir
import xml.etree.ElementTree as ET
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import average_precision_score
from numpy import mean, zeros
import numpy as np
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
#import _pickle as Pikcle
import json
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

def search_adaboost():
    fw = open('results1.txt', 'a')
    fw.write("Ada Boost: rf = AdaBoostClassifier(random_state=0,base_estimator=SVC(probability=True,kernel='poly')"
             " Parameters:n_estimators: [2,3]" + '\n')
    avg_prec = []
    i = 0
    for object in classes:
        i += 1
        print(object + " " + str(i))
        fw.write(str(object) + " " + str(i)+'\n')
        X, Y = make_binary_dataset(features, labels, object)
        rf = AdaBoostClassifier(random_state=0,base_estimator=SVC(probability=True,kernel='poly'))
        parameters = {'n_estimators': [2,3]}
        clf = GridSearchCV(rf, parameters, scoring='average_precision', verbose=1, n_jobs=4, cv=2)
        clf.fit(X, Y)
        print(clf.best_params_)
        fw.write(json.dumps(clf.best_params_)+'\n')
        print(clf.best_score_)
        fw.write(str(clf.best_score_)+'\n')
        avg_prec.append(clf.best_score_)
        X_test,Y_test=make_binary_dataset(features_test,labels_test,object)
        Y_pred=clf.predict(X_test)
        score=average_precision_score(Y_test,Y_pred)
        print("Average Precision Score:",score)
        temp = Y_test == Y_pred
        accuracy = (np.sum(temp) / len(Y_pred))
        print("Accuracy:", accuracy)
        fw.write("Average Precision Score:" + str(score) + '\n')
        fw.write("Accuracy:" + str(accuracy) + '\n')

    print(mean(avg_prec))
    fw.write(str(mean(avg_prec))+'\n')
    fw.close()

def search_adaboost1():
    fw = open('results1.txt', 'a')
    fw.write("Ada Boost: Parameters:n_estimators: [300]" + '\n')
    avg_prec = []
    avg_prec1= []
    i = 0
    for object in classes:
        i += 1
        print(object + " " + str(i))
        fw.write(str(object) + " " + str(i)+'\n')
        X, Y = make_binary_dataset(features, labels, object)
        rf = AdaBoostClassifier(random_state=0)
        parameters = {'n_estimators': [300]}
        clf = GridSearchCV(rf, parameters, scoring='average_precision', verbose=1, n_jobs=4, cv=2)
        clf.fit(X, Y)
        print(clf.best_params_)
        fw.write(json.dumps(clf.best_params_)+'\n')
        print(clf.best_score_)
        fw.write(str(clf.best_score_)+'\n')
        avg_prec.append(clf.best_score_)
        X_test,Y_test=make_binary_dataset(features_test,labels_test,object)
        Y_pred=clf.predict_proba(X_test)
        score=average_precision_score(Y_test,Y_pred[:,1])
        print("Average Precision Score:",score)
        temp = Y_test == Y_pred
        accuracy = (np.sum(temp) / len(Y_pred))
        print("Accuracy:", accuracy)
        fw.write("Average Precision Score:" + str(score) + '\n')
        fw.write("Accuracy:" + str(accuracy) + '\n')
        avg_prec1.append(score)

    print(mean(avg_prec))
    fw.write(str(mean(avg_prec))+'\n')
    print('Avg Precision Test:', mean(avg_prec1))
    fw.write('Avg Precision Test:' + str(mean(avg_prec1)))

    fw.close()


if __name__ == '__main__':
    image_annotations = "./Annotations/"
    image_annotations_test = "./Annotations_test/"
    features = load_features("train_features_fc.txt")
    labels, classes = load_labels(image_annotations)
    features = normalize(features)
    features_test=load_features("test_features_fc.txt")
    features_test=normalize(features_test)
    labels_test, classes_test = load_labels(image_annotations_test)
    #search_adaboost()
    search_adaboost1()
    fw=open('results1.txt','a')
    fw.write("Random Forest: Parameters:n_estimators: [300], 'max_depth': [5,7,9]" + '\n')
    avg_prec = []
    avg_prec1 = []
    i = 0
    for object in classes:
        i += 1
        print(object + " " + str(i))
        fw.write(str(object) + " " + str(i) + '\n')
        X, Y = make_binary_dataset(features, labels, object)
        rf = RandomForestClassifier(random_state=0)
        parameters = {'n_estimators': [300], 'max_depth': [5,7,9]}
        clf = GridSearchCV(rf, parameters, scoring='average_precision', verbose=1, n_jobs=4, cv=2)
        clf.fit(X,Y)
        print(clf.best_params_)
        fw.write(json.dumps(clf.best_params_)+'\n')
        print(clf.best_score_)
        fw.write(str(clf.best_score_)+'\n')
        avg_prec.append(clf.best_score_)
        X_test,Y_test=make_binary_dataset(features_test,labels_test,object)
        Y_pred=clf.predict_proba(X_test)
        score=average_precision_score(Y_test,Y_pred[:,1])
        print("Average Precision Score:",score)
        temp=Y_test==Y_pred
        accuracy=(np.sum(temp)/len(Y_pred))
        print("Accuracy:",accuracy)
        fw.write("Average Precision Score:"+str(score)+'\n')
        fw.write("Accuracy:" + str(accuracy) + '\n')
        avg_prec1.append(score)

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
    fw.write(str(mean(avg_prec)))
    print('Avg Precision Test:',mean(avg_prec1))
    fw.write('Avg Precision Test:'+str(mean(avg_prec1)))

    fw.close()

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