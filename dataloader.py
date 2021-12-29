import os
import cv2
import numpy as np
from tqdm import tqdm 

class DataLoader:

    def __init__(self, path, batch_size, classes=None):
        self.batch_size = batch_size
        self.path = path
        self.classes = classes if classes else os.listdir(path)
        self.classes = {index:_class for index,_class in enumerate(self.classes)}

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
                if counter%250==0:
                    break
                img_path = os.path.join(_path, image)
                img = cv2.resize(cv2.imread(img_path, 0),(128, 128))
                img = img.flatten()/255.0
                images.append(img)
                labels.append(_class_index)
                counter+=1
        images = np.array(images)
        labels = np.array(labels)
        print(images.shape)
        
        return (images, labels)
            

    
    def preprocess(self):
        for _class in self.classes:
            _path = os.path.join(self.path, _class)
            print(f"number of pictures belongs to {_class} -> {len(os.listdir(_path))}")