import tensorflow as tf
import tables
import matplotlib.pyplot as plt
from tensorflow.contrib.keras import models
import numpy as np
from tensorflow.contrib.keras import layers, losses, optimizers, metrics, activations
from sklearn.preprocessing import MinMaxScaler
from generate_data import retrieve_data


_, t, _ = retrieve_data(False, False)
# open and read
# hdf5_file = tables.open_file('frogtrainingdata.hdf5', mode='r')
# index = 99
# E = hdf5_file.root.E_real[index, :] + 1j * hdf5_file.root.E_imag[index, :]
# fig, ax = plt.subplots(2, 1)
# ax[0].pcolormesh(hdf5_file.root.frog[index, :].reshape(57, 334), cmap='jet')
# ax[1].plot(t, np.real(E), color='blue')
# ax[1].plot(t, np.imag(E), color='red')
# ax[1].plot(t, np.abs(E), color='blue', linestyle='dashed')
# # plt.show()
# hdf5_file.close()

model = models.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(57, 334, 1)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
# model.add(layers.Dense(2400))
model.add(layers.Dense(40))
# model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.2))
# model.add(layers.Dense(1200))
model.add(layers.Dense(1200))

model.compile(optimizer='adam', loss='mean_squared_error')


hdf5_file = tables.open_file('frogtrainingdata.hdf5', mode='r')
index = 99
E_real = hdf5_file.root.E_real[:, :]
E_imag = hdf5_file.root.E_imag[:, :]
frog = hdf5_file.root.frog[:, :]
scaler_E = MinMaxScaler()
scaler_frog = MinMaxScaler()
E_apended = np.concatenate((E_real, E_imag), 1)
# E_scaled = scaler_E.fit_transform(E_apended)
# frog_scaled = scaler_frog.fit_transform(frog)
hdf5_file.close()


model.fit(frog.reshape(-1, 57, 334, 1), E_apended, epochs=1)

# check with a value from dataset
index_test = 1

E_actual = E_real[index_test] + 1j * E_imag[index_test]
pred = model.predict(frog[index_test].reshape(1, 57, 334, 1))
print(pred.shape)
E_imag_pred = pred[0][600:]
E_real_pred = pred[0][:600]
E_pred = E_real_pred + 1j * E_imag_pred

fig, ax = plt.subplots(2, 2)
ax[0][0].pcolormesh(frog[index_test].reshape(57, 334), cmap='jet')
ax[0][1].plot(t, np.real(E_pred), color='blue')
ax[0][1].plot(t, np.imag(E_pred), color='red')
ax[0][1].plot(t, np.abs(E_pred), color='blue', linestyle='dashed')

ax[1][0].pcolormesh(frog[index_test].reshape(57, 334), cmap='jet')
ax[1][1].plot(t, np.real(E_actual), color='blue')
ax[1][1].plot(t, np.imag(E_actual), color='red')
ax[1][1].plot(t, np.abs(E_actual), color='blue', linestyle='dashed')


plt.show()


