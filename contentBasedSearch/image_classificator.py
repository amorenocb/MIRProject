import cv2
import sys, os
import numpy as np
import glove
import pickle
from pyflann import *

def getImageClasses(synset_words):
    # First of all we will create a dictionary of the classes ID's, where the value fields
    # will contain the list of corresponding tags. For example:
    # {'n01440764'} : ['tench', 'Tinca tinca']
    classes = dict()
    class_order = []
    rows = open(synset_words).read().strip().split("\n")
    for row in rows:
        class_delimiter = row.find(" ")
        class_id = row[:class_delimiter]
        class_tags = [ tag.strip() for tag in row[class_delimiter:].split(",") ]
        classes[class_id] = class_tags
        class_order.append(class_id)

    return classes,class_order

def classifyPhotos(photos_dir, net_proto, net_model, top_n, synset_words, glove_instance):
    preprocessed_photos = []
    print("------- PRE-PROCESSING PHOTOS ------")
    i = 1
    for image_file in os.listdir(photos_dir):
        if(i % 200 == 0):
            print("PRE-PROCESSED {} PHOTOS SO FAR".format(i))
        if image_file.endswith(".jpg") or image_file.endswith(".jpeg"):
            image = cv2.imread(photos_dir+"/"+image_file)
            blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
            preprocessed_photos.append((blob,image_file))
        i+=1

    net = cv2.dnn.readNetFromCaffe(net_proto, net_model)
    classes, class_order = getImageClasses(synset_words)
    tag_photo_hash_table = dict()

    print("------- EXTRACTING FEATURES ------")
    i = 1
    for blob,image_name in preprocessed_photos:
        if (i % 200 == 0):
            print("PROCESSED {} PHOTOS SO FAR".format(i))
        net.setInput(blob)
        preds = net.forward()
        idxs = np.argsort(preds[0])[::-1][:top_n]
        for idx in idxs:
            sum_of_words = np.zeros(50)
            for tag in classes[class_order[idx]]:
                tag_vector = glove_instance.get_word_vector(tag)
                if tag_vector is not None:
                    sum_of_words += np.array(tag_vector)
            average_word = sum_of_words / len(classes[class_order[idx]])
            most_similar_terms = glove_instance._similarity_query(average_word, 2)
            for (word,sim) in most_similar_terms:
                tag_photo_hash_table.setdefault(word, []).append(image_name)
        i += 1

    with open("tags_images.txt", "wb") as myFile:
        pickle.dump(tag_photo_hash_table, myFile)

    return tag_photo_hash_table

if(len(sys.argv)>1):
    if(sys.argv[1] == 'classify'):
        top_n = int(sys.argv[2])
        photos_dir = sys.argv[3]
        net_proto = sys.argv[4]
        net_model = sys.argv[5]
        sys_word_file = sys.argv[6]
        glove_ins = glove.Glove()
        print("Loading Glove vectors")
        glove_instance = glove_ins.load_stanford(filename="vectorsGloveLight.txt")
        print("Finished loading Glove vectors")
        classifyPhotos(photos_dir,net_proto,net_model,top_n,sys_word_file, glove_instance)





