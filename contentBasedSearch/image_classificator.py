import cv2
import sys, os
import numpy as np

top_n = int(sys.argv[1])
photos_dir = sys.argv[2]
net_proto = sys.argv[3]
net_model = sys.argv[4]
sys_word_file = sys.argv[5]

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

def classifyPhotos(photos_dir, net_proto, net_model, top_n, synset_words):
    preprocessed_photos = []
    for image_file in os.listdir(photos_dir):
        if image_file.endswith(".jpg") or image_file.endswith(".jpeg"):
            image = cv2.imread(photos_dir+"/"+image_file)
            blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
            preprocessed_photos.append((blob,image_file))

    net = cv2.dnn.readNetFromCaffe(net_proto, net_model)
    classes, class_order = getImageClasses(synset_words)

    tag_photo_hash_table = dict()

    for blob,image_name in preprocessed_photos:
        net.setInput(blob)
        preds = net.forward()
        idxs = np.argsort(preds[0])[::-1][:top_n]
        for idx in idxs:
            tag_photo_hash_table.setdefault(classes[class_order[idx]][0],[]).append(image_name)

    return tag_photo_hash_table

classifyPhotos(photos_dir,net_proto,net_model,top_n,sys_word_file)





