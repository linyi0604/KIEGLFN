from aip import AipFace
import cv2
import math
import base64
import os
import json

# baidu face detection api settings
APP_ID = ''
API_KEY = ''
SECRET_KEY = ''
client = AipFace(APP_ID, API_KEY, SECRET_KEY)

load_path = "../../data/ACNE04/image_origin/"
save_path = "../../data/ACNE04/image_face_detection/ "
face_location={}
for file in os.listdir(load_path):
    try:
        img_path = load_path + file
        with open(img_path, "rb") as fp:
            base64_data = base64.b64encode(fp.read())
        image = str(base64_data, 'utf-8')
        imageType = "BASE64"
        client.detect(image, imageType)
        options = {}
        options["face_field"] = "age"
        options["max_face_num"] = 1
        options["face_type"] = "LIVE"
        result = client.detect(image, imageType, options)
        num=result['result']['face_num']
        face_location[file] = result["result"]['face_list'][0]['location']

        img=cv2.imread(load_path + file)
        if result['error_code'] == 'SDK108':
            with open('', 'w')as f:
                f.write(file+'\n')
        else:
            if result ['result']==None:
                cv2.imwrite(save_path + file, img)
            else:
                num = result['result']['face_num']
                location = result['result']['face_list'][num-1]['location']
                Theta = location['rotation'] / 60
                A = (int(location['left']) + int(location['height'] * math.sin(Theta) * 0.45),
                     int(location['top']) - int(location['height'] * math.cos(Theta) * 0.45))
                B = (int(location['left']) + int(location['width'] * math.cos(Theta)) + int(
                    location['height'] * math.sin(Theta) * 0.45),
                     int(location['top']) + int(location['width'] * math.sin(Theta)) - int(
                         location['height'] * math.cos(Theta) * 0.45))
                AC_Len = math.sqrt(location['width'] ** 2 + location['height'] ** 2)
                AC_Theta = math.atan(location['height'] / location['width']) +location['rotation'] / 60
                C = (int(location['left']) + int(AC_Len * math.cos(AC_Theta)),
                     int(location['top']) + int(AC_Len * math.sin(AC_Theta)))
                D = (int(location['left']) - int(location['height'] * math.sin(Theta)),
                     int(location['top']) + int(location['height'] * math.cos(Theta)))
                #top:
                if A[1]>=B[1]:
                    up=B[1]
                else:
                    up=A[1]
                if up<=0:
                    up=0
                else:
                    pass
                #left:
                left=int((A[0]+D[0])*0.5)
                if left<=0:
                    left=0
                else:
                    pass
                #below
                if C[1] >= D[1]:
                    below = C[1]
                else:
                    below = D[1]
                if below >= img.shape[0]:
                    below = img.shape[0]
                else:
                    pass
                #right
                right=int((B[0]+C[0])*0.5)
                if right >= img.shape[1]:
                    right = img.shape[1]
                else:
                    pass
                cropped = img[up:below,left:right]
                cv2.imwrite(save_path + file, cropped)

    except:
        with open('timeout.txt', 'w')as f2:
            f2.write(file+'\n')


face_location = json.dumps(face_location,ensure_ascii=False,indent=5)
with open('face_location.json', 'w', encoding='utf-8') as w:
    w.write(face_location)
#
#
