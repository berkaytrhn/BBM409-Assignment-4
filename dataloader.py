import os
import cv2
import numpy as np
from tqdm import tqdm 

class DataLoader:

    def __init__(self, path, classes=None):
        self.path = path
        self.classes = classes if classes else os.listdir(path)
        print(self.classes)

    def load_images(self):
        # avg height -> 252.63016157989227
        # avg width  -> 320.0388097329921
        img_sizes = []
        for _class in tqdm(self.classes):
            _path = os.path.join(self.path, _class)
            for image in tqdm(os.listdir(_path)):
                img_path = os.path.join(_path, image)
                img = cv2.imread(img_path)
            
        img_sizes = np.array(img_sizes)
        print("avg height: ", np.mean(img_sizes[:,0]))
        print("avg width: ", np.mean(img_sizes[:,1]))

    
    def preprocess(self):
        for _class in self.classes:
            _path = os.path.join(self.path, _class)
            print(f"number of pictures belongs to {_class} -> {len(os.listdir(_path))}")