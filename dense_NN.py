import pickle
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

x, y = make_classification(10000, 1000)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9)

model = Sequential()
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model_checkpoint = ModelCheckpoint('NN.{epoch:02d}-{val_accuracy:.2f}.hdf5')
model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test),
          callbacks=[model_checkpoint])

training_predictions = model.predict(x_train)
testing_predictions = model.predict(x_test)

file = open('NN_training_predictions.txt', 'wb')
pickle.dump(training_predictions, file)
file.close()

file = open('NN_testing_predictions.txt', 'wb')
pickle.dump(testing_predictions, file)
file.close()
