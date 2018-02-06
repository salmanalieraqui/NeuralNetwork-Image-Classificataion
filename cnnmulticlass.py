from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


classifier = Sequential()

#first  hidden layer.....
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#second hidden layer
classifier.add(Convolution2D(64,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#second hidden layer...
#classifier.add(Convolution2D(128,3,3,activation='relu'))
#classifier.add(MaxPooling2D(pool_size=(2,2)))

#flatting....
classifier.add(Flatten())

#fully connected layer 1
classifier.add(Dense(output_dim=128,activation='relu'))

#fully connected layer 2
#classifier.add(Dense(output_dim=128,activation='relu'))

#fully connected layer 3
#classifier.add(Dense(output_dim=128,activation='relu'))

#fully connected layer 1
classifier.add(Dense(output_dim=128,activation='relu'))

#output layer
classifier.add(Dense(output_dim=3,activation='softmax'))

#loss fucntion.
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])



#daatagen initilization..
train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)




training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='categorical')

classifier.fit_generator(training_set,
                    samples_per_epoch=802,
                    epochs=10,
                    validation_data=test_set,
                    nb_val_samples=226)
