import os
import numpy as np 
import pandas as pd 
from scipy.misc import imread
import keras
from keras.layers import Dense, Flatten, Reshape, InputLayer
from keras.regularizers import L1L2
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
import matplotlib.pyplot as plt

seed = 128
rng = np.random.RandomState(seed)

# Path of the data
root_dir = os.path.abspath('.')



# Loading the data 
train = pd.read_csv(os.path.join(root_dir,'Train','train.csv'))
test = pd.read_csv(os.path.join(root_dir,'test.csv'))

temp = []
for img_name in train.filename:
	image_path  = os.path.join(root_dir,'Train','Images','train',img_name)
	img = imread(image_path,flatten = True)
	img = img.astype('float32')
	temp.append(img)

train_X = np.stack(temp) # Stack all the lists one below the other

train_X = train_X /255 # Divide each value in the list by 255


# Data visualisation
img_name = rng.choice(train.filename)
filepath = os.path.join(root_dir,'Train','Images','train',img_name)

img = imread(filepath,flatten = True)
plt.imshow(img,cmap = 'gray')
plt.show()

g_input_shape = 100 # Generator 
d_input_shape = (28,28) # Discriminator
hidden_layer_1 = 500
hidden_layer_2 = 500
g_output_layer = 784 # 28*28 
d_output_layer = 1 
epochs = 25
batch_size = 128 

# Generator Model 
model_g = Sequential([
    Dense(units=hidden_1_num_units, input_dim=g_input_shape, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)),

    Dense(units=hidden_2_num_units, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)),
        
    Dense(units=g_output_num_units, activation='sigmoid', kernel_regularizer=L1L2(1e-5, 1e-5)),
    
    Reshape(d_input_shape),
])

# discriminator
model_d = Sequential([
    InputLayer(input_shape=d_input_shape),
    
    Flatten(),
        
    Dense(units=hidden_1_num_units, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)),

    Dense(units=hidden_2_num_units, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)),
        
    Dense(units=d_output_num_units, activation='sigmoid', kernel_regularizer=L1L2(1e-5, 1e-5)),
])

# Compiling the GAN 

gan = simple_gan(model_g,model_d,normal_latent_sampling((100,)))
model = AdversarialModel(base_model=gan,player_params=[model_g.trainable_weights, model_d.trainable_weights])
model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(), player_optimizers=['adam', 'adam'], loss='binary_crossentropy')

# Training the GAN 
res = model.fit(x = train_X,y = gan_targets(train_X.shape[0]),epochs = 10, batch_size = batch_size)

# Visualisation 
plt.plot(history.history['player_0_loss'])
plt.plot(history.history['player_1_loss'])
plt.plot(history.history['loss'])


# Generation of the image   

samples = np.random.normal(size = (10,100))
pred = model_g.predict(samples)
for i in range(pred.shape[0]):
	plt.imshow(pred[i,:],cmap = 'gray')
	plt.show()