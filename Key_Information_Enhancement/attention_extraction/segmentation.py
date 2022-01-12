import os
import PIL.Image as Image
import numpy as np
from sklearn.cluster import KMeans


path = "../../data/ACNE04/image_face_detection/"
save_path = "../../data/ACNE04/image_attention/"
img_names = os.listdir(path)


def get_key_cluster(color_std, center):
    c1 = center[0]
    c2 = center[1]
    s1 = np.mean(np.abs(c1-color_std))
    s2 = np.mean(np.abs(c2-color_std))
    if s1 > s2:
        return 1
    else:
        return 0


for i, imgname in enumerate(img_names):
    filePath = path+imgname
    f = open(filePath,'rb')
    data = []
    img = Image.open(f)
    m,n = img.size#the size of image

    w, h = img.size
    wmid = w//2
    hmid = h//2
    crop = img.crop((wmid-1, hmid-1, wmid+2, hmid+2))
    color = []
    for i in range(3):
        for j in range(3):
            r, g, b = crop.getpixel((i, j))
            color.append((r, g, b))
    color_mean = np.mean(color, axis=0)

    max_gap = np.array([255, 255, 255]) - color_mean
    min_gap = color_mean
    for i in range(m):
        for j in range(n):
            x,y,z = img.getpixel((i,j))
            pixel = np.array([x,y,z])
            pixel = pixel  - color_mean
#             pixel = np.array([x,y,z])
            if np.sum(pixel) < 0:
#                 pixel /= min_gap
                pixel = np.true_divide(pixel,min_gap)
            else:
#                 pixel /= max_gap
                pixel = np.true_divide(pixel,max_gap)
            # pixel = np.abs(pixel)
            data.append(pixel)
    f.close()
    imgData = np.mat(data)


    row, col = m, n
    img_origin = img
    km = KMeans(n_clusters=2)
    label = km.fit_predict(imgData)
    #get the label of each pixel
    label = label.reshape([row,col])
    #create a new image to save the result of K-Means
    pic_new = Image.new("RGB",(row,col))
    #according to the label to add the pixel

    key = get_key_cluster((0,0,0), km.cluster_centers_)
    for i in range(row):
        for j in range(col):
            if label[i][j] == key:
                pixel = img_origin.getpixel((i, j))
            else:
                pixel = (0, 0, 0)
            pic_new.putpixel((i,j),pixel)
    pic_new.save(save_path+imgname,"JPEG")
