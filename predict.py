import tensorflow as tf
import numpy as np
from utils import load_dataset
import selectivesearch
import cv2
import os

classes=['air-hole', 'hollow-bead', 'slag-inclusion', 'bite-edge', 'broken-arc', 'crack', 'overlap', 'unfused']
colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(255,255,255),(0,0,0)]
min_w,max_w,min_h,max_h=7, 854, 6, 791
image_width,image_height=50,50
# (train_labels,train_images),(test_labels,test_images)=load_dataset(resize=(image_width,image_height))
# train_images=np.expand_dims(train_images,axis=3)
# test_images=np.expand_dims(test_images,axis=3)

model = tf.keras.models.load_model('./model.h5')
model.summary()
# loss, acc = model.evaluate(test_images,test_labels,verbose=2)
# print('loss:',loss,'acc:',acc)
images_name=[]
for home, dirs, files in os.walk('./images/'):
    for filename in files:
        images_name.append(filename)

for image_name in images_name:
    image=cv2.imread('./images/'+image_name)
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # image_gray = cv2.fastNlMeansDenoising(image_gray,None,10,7,21)

    img_lbl,regions=selectivesearch.selective_search(image, scale=600, sigma=0.9, min_size=110)
    print('selective_search_result:',len(regions))

    cut_images=[]
    cut_images_coord=[]
    for region in regions:
        x,y,w,h=region['rect']
        if(w>min_w and w<max_w and h>min_h and h<max_h and x>20 and x<image_gray.shape[1]-20 and y>20 and y<image_gray.shape[0]):
            cut_images_coord.append([x,y,w,h])
            cut_images.append(cv2.resize(image_gray[y:y+h,x:x+w],(image_width,image_height)))

    cut_images=np.array(cut_images,dtype=np.float)
    cut_images/=255.0
    cut_images=np.expand_dims(cut_images,axis=3)
    out=model.predict(cut_images)
    # print(out.shape)

    for i in range(len(out)):
        x,y,w,h=cut_images_coord[i]
        image=cv2.rectangle(image, (x, y), (x+w, y+h), colors[np.argmax(out[i])], 1, lineType=cv2.LINE_AA)
        cv2.putText(image,classes[np.argmax(out[i])]+' '+str(np.max(out[i])),(x,y),cv2.FONT_HERSHEY_DUPLEX,0.5,colors[np.argmax(out[i])],thickness=1,lineType=cv2.LINE_AA)

    cv2.imwrite('./result/'+image_name,image)
