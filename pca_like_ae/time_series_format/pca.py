from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import scipy.io
import random


data = np.array(scipy.io.loadmat('timeseries_midi_dataset_with_transpose.mat')['train_data'], dtype=np.float32)
shape_data = data.shape                                                               # store shape of y before reshaping it
data = np.reshape(data,[shape_data[0],shape_data[1]*shape_data[2]])/255 

X_train, X_test =train_test_split(data, test_size=0.05)

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def AnalyticalPCA(y, dimension):
    pca = PCA(n_components=dimension)
    pca.fit(y)
    return pca
                                                          # store shape of y before reshaping it

p_analytical = AnalyticalPCA(X_train,150)  

test_eval = {'known' : {}, 'new': {}}
print("Testing known data")
for x in range(5):
    rand_index = np.random.randint(X_train.shape[0])
    known_data = X_train[rand_index]
    
    latent_space = p_analytical.transform([known_data])
    modified_components = [i+random.uniform(-0.2,0.2) for i in latent_space[0][:20]]
    latent_space_2 = np.array([np.concatenate((modified_components, latent_space[0][20:]))])
    reconstructed_x = p_analytical.inverse_transform(latent_space)
    reconstructed_x = np.reshape(reconstructed_x*255, (128, 4))
    reconstructed_x = np.rint(reconstructed_x)
    real_data = np.reshape(known_data*255, (128, 4))

    reconstructed_random = p_analytical.inverse_transform(latent_space_2)
    reconstructed_random = np.reshape(reconstructed_random*255, (128, 4))
    reconstructed_random = np.rint(reconstructed_random)

    test_eval['known'][x] = {'input': real_data, 'latent_space': latent_space, 'output': reconstructed_x, 'mae': mae(real_data,reconstructed_x), 'random': reconstructed_random}


print("Testing new data")
for x in range(5):
    rand_index = np.random.randint(X_test.shape[0])
    new_data = X_test[rand_index]

    
    latent_space = p_analytical.transform([new_data])
    reconstructed_x = p_analytical.inverse_transform(latent_space)
    reconstructed_x = np.reshape(reconstructed_x*255, (128, 4))
    reconstructed_x = np.rint(reconstructed_x)
    real_data = np.reshape(new_data*255, (128, 4))

    test_eval['new'][x] = {'input': real_data, 'latent_space': latent_space, 'output': reconstructed_x, 'mae': mae(real_data,reconstructed_x)}

print("Testing random")


with open('pcaae_models/pca/evaluation_pca.pickle', 'wb') as handle:
        pickle.dump(test_eval, handle, protocol=pickle.HIGHEST_PROTOCOL)


# print(latent_space)

# reconstructed_x = p_analytical.inverse_transform(latent_space)# reconstruye "x" desde el espacio latente
# # print(reconstructed_x.shape)
# reconstructed_x = np.reshape(reconstructed_x*255, (128, 4)) # reformatea el array reconstruido a su forma original (128,4)
# real_data = np.reshape(rand_data*255, (128, 4))
# # print((reconstructed_x * 255).astype(int))

# # Verify that the reconstruction error is small
# # error = np.linalg.norm(rand_data*255  - (reconstructed_x*255).astype(int), ord='fro')
# # print("Reconstruction error:", error)


# print(real_data[:10])

# print((reconstructed_x).astype(int)[:10])


# print(mae(real_data,reconstructed_x))

