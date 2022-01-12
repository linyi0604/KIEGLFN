import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import json


path = "../../data/ACNE04/image_face_detection/"
image_name = os.listdir(path)
sc_dict = {}
path_size = [i for i in range(1, 22, 2)]

def crop_center_n(img, n):
    h, w = img.size
    midh, midw = h//2, w//2
    size = n // 2
    patch = img.crop((midw-size, midh-size, midw+size+1, midh+size+1))
    return patch
def patch_channel_mean(patch):
    patch = np.array(patch).reshape([-1, 3])
    mv = np.mean(patch, axis=0)
    return mv


for name in image_name:
    img = Image.open(path + name)
    img = img.resize((255, 255))
    sc_list = []
    for n in path_size:

        patch = crop_center_n(img, n)
        skin_mean = patch_channel_mean(patch)
        max_gap = np.array([255, 255, 255]) - skin_mean
        min_gap = skin_mean
        data_img = []
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                x,y,z = img.getpixel((i, j))
                pixel = np.array((x,y,z))
                pixel = pixel - skin_mean
                if np.sum(pixel) < 0:
                    pixel = np.true_divide(pixel, min_gap)
                else:
                    pixel = np.true_divide(pixel, max_gap)
                data_img.append(pixel)
        data_img = np.mat(data_img)
        km = KMeans(n_clusters=2)
        label = km.fit_predict(data_img)
        sc = silhouette_score(data_img, label)
        sc_list.append((n, sc))
        print(name, n, sc)
    sc_dict[name] = sc_list
    print(sc_list)

with open("./Silhouette_Coefficient/sc_dict.pkl", "wb") as f:
    pickle.dump(sc_dict, f)
with open("./Silhouette_Coefficient/sc_dict.json", "w") as f:
    json.dump(sc_dict, f, indent=2)
