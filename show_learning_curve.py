import matplotlib.pyplot as plt
import numpy as np

# 학습 곡선 그리기
# num_epochs = len(history.history['accuracy'])
epoch_range = np.arange(1, num_epochs + 1)

# Combined n_loss, i_loss, val_n_loss, val_i_loss, and val_kid
plt.plot(epoch_range, history.history['n_loss'], label='n_loss')
plt.plot(epoch_range, history.history['i_loss'], label='i_loss')
plt.plot(epoch_range, history.history['val_n_loss'], label='val_n_loss')
plt.plot(epoch_range, history.history['val_i_loss'], label='val_i_loss')
plt.plot(epoch_range, history.history['val_kid'], label='val_kid')
plt.title('Overall Model Losses & KID')
plt.ylabel('Values')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Combined n_loss and i_loss
plt.plot(epoch_range, history.history['n_loss'], label='n_loss')
plt.plot(epoch_range, history.history['i_loss'], label='i_loss')
plt.title('Model Train Noise Loss & Image Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Model Performance
plt.plot(epoch_range, history.history['val_n_loss'], label='val_n_loss')
plt.plot(epoch_range, history.history['val_i_loss'], label='val_i_loss')
plt.plot(epoch_range, history.history['val_kid'], label='val_kid')
plt.title('Model Performance')
plt.ylabel('Performance')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.ylim([0.0, 1.0])
plt.show()

# Model Train Noise Loss
plt.plot(epoch_range, history.history['n_loss'])
plt.title('Model Train Noise Loss')
plt.ylabel('Noise Loss')
plt.xlabel('Epoch')
plt.legend(['n_loss'], loc='upper left')
plt.show()

# Model Train Image Loss
plt.plot(epoch_range, history.history['i_loss'])
plt.title('Model Train Image Loss')
plt.ylabel('Image Loss')
plt.xlabel('Epoch')
plt.legend(['i_loss'], loc='upper left')
plt.show()

# Model Valid Noise Loss
plt.plot(epoch_range, history.history['val_n_loss'])
plt.title('Model Valid Noise Loss')
plt.ylabel('Noise Loss')
plt.xlabel('Epoch')
plt.legend(['val_n_loss'], loc='upper left')
plt.show()

# Model Valid Image Loss
plt.plot(epoch_range, history.history['val_i_loss'])
plt.title('Model Valid Image Loss')
plt.ylabel('Image Loss')
plt.xlabel('Epoch')
plt.legend(['val_i_loss'], loc='upper left')
plt.show()

# Model KID
plt.plot(epoch_range, history.history['val_kid'])
plt.title('Model KID')
plt.ylabel('KID')
plt.xlabel('Epoch')
plt.legend(['val_kid'], loc='upper left')
plt.show()
