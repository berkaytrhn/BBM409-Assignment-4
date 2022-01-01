import os
import cv2
import numpy as np
from tqdm import tqdm 
from sklearn.model_selection import train_test_split

class DataLoader:

    def __init__(self, path, size, classes=None):
        self.path = path
        self.classes = classes if classes else os.listdir(path)
        self.classes = {index:_class for index,_class in enumerate(self.classes)}
        self.img_height, self.img_width=size

        print(self.classes)

    def load_images(self):
        # avg height -> 252.63016157989227
        # avg width  -> 320.0388097329921
        images = []
        labels = []
        for _class_index in tqdm(self.classes.keys()):
            _class = self.classes[_class_index]
            counter=1
            _path = os.path.join(self.path, _class)
            for image in tqdm(os.listdir(_path)):
                """
                if counter%30==0:
                    break
                """
                img_path = os.path.join(_path, image)
                img = cv2.resize(cv2.imread(img_path, 0),(self.img_height, self.img_width))
                img = img.flatten()/255.0
                images.append(img)
                labels.append(_class_index)
                counter+=1
        images = np.array(images)
        labels = np.array(labels)
        print(images.shape)
        
        return (images, labels)
    
    def batch_and_split_process(self, images, labels, batch_size, test_size=0.2, valid_size=0.2):
        self.batch_size=batch_size
        # test set
        X_train, X_test, y_train, y_test = train_test_split(images, labels, shuffle=True, test_size=test_size)

        # validation set
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, shuffle=True, test_size=valid_size)
        

        sets = [X_train, y_train]

        batched_sets = []

        for _set in sets:
            _shape = _set.shape
            _step = _shape[0]//self.batch_size
            batched=[]
            for i in range(_step+1):
                batched.append(_set[i*self.batch_size:(i+1)*self.batch_size])
            batched = np.array(batched)
            #print(batched.shape)
            batched_sets.append(batched)
            """
            for i in batched:
                print("----")
                print(i.shape)
                print(i)
            """
        batched_sets = np.array(batched_sets)
        return np.array([batched_sets[0], X_valid, X_test, batched_sets[1], y_valid, y_test])