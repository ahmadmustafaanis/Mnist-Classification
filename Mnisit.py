from keras.datasets import mnist
from keras import Sequential,optimizers,layers,models
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(training_data,training_labels),(testing_data,testing_labels)=mnist.load_data()
#seeing sample data
plt.imshow(training_data[0])
plt.show()

#normalizing data

training_data = training_data.reshape((60000,28*28)).astype('float32')/255 #/255 to get values b/w 0 and 1
testing_data= testing_data.reshape((10000,28*28)).astype('float32')/255

training_labels = to_categorical(training_labels)
testing_labels = to_categorical(testing_labels)

#creating a model
model = models.Sequential()
#our model is going to have 1 input unit with 28*28 nodes, 1 hidden unit with 512 nodes, and 1 output unit with 10 nodes
model.add(layers.Dense(512,activation='relu',input_shape = (28*28,)))
model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer = optimizers.Adam(lr=0.01),loss='mse',metrics=['accuracy'])
model.fit(training_data,training_labels,batch_size=128,epochs=20)

test_loss,test_acc = model.evaluate(training_data,training_labels)

print("TESTING ACCURACY: ",test_acc)