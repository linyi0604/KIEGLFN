import cv2
import os

image_path = "../../data/ACNE04/image_origin/"
save_path = "./image_compression/"
x=0
for root,dirs,files in os.walk(image_path):
    for file in files:
        print(file)
        img_path = root + '/' + file
        #img_path =r'C:\Users\24106\Desktop\go\levle1_257.jpg'
        print(img_path)
        img = cv2.imread(img_path)
        #cv2.imshow('img', img)
        #print(img.shape)
        x = x + 1
        img_saving_path = save_path
        cv2.imwrite(img_saving_path+'/'+file,img,[cv2.IMWRITE_JPEG_QUALITY,30])







# from aip import AipFace
# import cv2
# import matplotlib.pyplot as plt
# import math
# import base64
# import os
#
# """ 你的 APPID AK SK """
# APP_ID = '23068362'
# API_KEY = 'epDx92ilufUguH7SUvdrkQ31'
# SECRET_KEY = 'hKdY9Hbfbuplj4RmVxDmvhVqPMap13BY'
#
# #client = AipFace(APP_ID, API_KEY, SECRET_KEY)
# for root,dirs,files in os.walk('C:/Users/24106/Desktop/NEW JPEGImages_rotateup'):
#     for file in files:
#         client = AipFace(APP_ID, API_KEY, SECRET_KEY)
#         print(file)
#         img_path = root + '/' + file
#         img_saving_path = r'C:/Users/24106/Desktop/RE JPEGImages_rotateup'
#         with open(img_path, "rb") as fp:
#             base64_data = base64.b64encode(fp.read())
#         image = str(base64_data, 'utf-8')
#         imageType = "BASE64"
#         client.detect(image, imageType)
#
#         options = {}
#         options["face_field"] = "age"
#         options["max_face_num"] = 1
#         options["face_type"] = "LIVE"
#         result=client.detect(image, imageType, options)
#         print(result)
#         img=cv2.imread('C:/Users/24106/Desktop/JPEGImages_rotateup'+'/'+file)
#         num = result['result']['face_num']
#         if num !=0:
#             location = result['result']['face_list'][num-1]['location']
#             Theta = location['rotation'] / 60
#
# #----------------------------------------------
# #"""只有脸部"""
# #----------------------------------------------
#             A = (int(location['left']),int(location['top']))
#             B = (int(location['left'])+int(location['width']*math.cos(Theta)),int(location['top'])+int(location['width']*math.sin(Theta)))
#             AC_Len = math.sqrt(location['width']**2 + location['height']**2)
#             AC_Theta = math.atan(location['height']/location['width'])+location['rotation']/60  ####或者是？？？
#             C = (int(location['left']) + int(AC_Len*math.cos(AC_Theta)), int(location['top'])+int(AC_Len*math.sin(AC_Theta)))
#             D = (int(location['left']) - int(location['height'] * math.sin(Theta)),
#                 int(location['top']) + int(location['height'] * math.cos(Theta)))
#
# #------------------------------------------------
# #"""直接取一半"""
# #------------------------------------------------
# # A = (int(location['left']) - int(location['height'] * math.sin(Theta)),0)
# # B = (int(location['left']) - int(location['height'] * math.sin(Theta)),size[0])
# # C = (size[1],size[0])
# # D = (size[1],0)
#
#
# #'''画框'''
#             cv2.line(img, A, B, (0, 0, 255), 2)
#             cv2.line(img, B, C, (0, 0, 255), 2)
#             cv2.line(img, C, D, (0, 0, 255), 2)
#             cv2.line(img, D, A, (0, 0, 255), 2)
#             cv2.imwrite(img_saving_path+'/'+file,img)
#
#
#
#
#
