from utils import load_dataset
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, metrics
import numpy as np
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

image_width,image_height=100,50
classes=['air-hole', 'hollow-bead', 'slag-inclusion', 'bite-edge', 'broken-arc', 'crack', 'overlap', 'unfused']

(train_labels,train_images),(test_labels,test_images)=load_dataset(resize=(image_width,image_height))
# (labels,images)=load_dataset(resize=(image_width,image_height),concat=True)
train_images=np.expand_dims(train_images,axis=3)
test_images=np.expand_dims(test_images,axis=3)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(image_height,image_width,1)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="training/cp-{epoch:03d}.ckpt",
                                                 monitor='val_loss',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=True)

history=model.fit(train_images, train_labels,batch_size=128, epochs=40,callbacks=[checkpoint],validation_data=(test_images,test_labels),shuffle=True)

test_loss,test_acc=model.evaluate(test_images,  test_labels, verbose=2)
print('test_loss:',test_loss,'test_acc:',test_acc)

model.save('./model.h5')

plt.subplot(1,2,1)
plt.plot(history.epoch,history.history.get('loss'),color='green',label = 'loss')
plt.plot(history.epoch,history.history.get('val_loss'),color='red',label = 'val_loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.epoch,history.history.get('acc'),color='green',label = 'acc')
plt.plot(history.epoch,history.history.get('val_acc'),color='red',label = 'val_acc')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
plt.legend()

plt.show()