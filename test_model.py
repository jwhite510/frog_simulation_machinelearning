from tensorflow.contrib.keras import models
import matplotlib.pyplot as plt
from generate_data import retrieve_data
import numpy as np

model = models.load_model('./2000_epoch_1000_sample/2000_epochs_1000_samples.hdf5')

E_actual, t, frogtrace_flat = retrieve_data(plot_frog_bool=False, print_size=False)


# plt.pcolormesh(frogtrace_flat.reshape(58, 106))
# plt.show()

# pred = model.predict(frog[index_test].reshape(1, 58, 106, 1))
pred = model.predict(frogtrace_flat.reshape(1, 58, 106, 1))

E_imag_pred = pred[0][128:]
E_real_pred = pred[0][:128]
E_pred = E_real_pred + 1j * E_imag_pred


fig, ax = plt.subplots(2, 2)
# ax[0][0].pcolormesh(frog[index_test].reshape(58, 106), cmap='jet')
ax[1][1].plot(t, np.abs(E_pred), color='blue', linestyle='dashed')
ax[1][1].plot(t, np.real(E_pred), color='blue')
ax[1][1].plot(t, np.imag(E_pred), color='red')

ax[0][0].pcolormesh(frogtrace_flat.reshape(58, 106), cmap='jet')
ax[0][1].plot(t, np.abs(E_actual), color='blue', linestyle='dashed')
ax[0][1].plot(t, np.real(E_actual), color='blue')
ax[0][1].plot(t, np.imag(E_actual), color='red')

plt.show()