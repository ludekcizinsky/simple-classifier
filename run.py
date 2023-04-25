import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Load images
paths = [i for i in range(1, 3)]
images = []
for i in paths:
  image = Image.open(f"images/img{i}.jpg")
  arr = np.asarray(image)
  images.append(arr)

# Turn the list of images (a.k.a. list of 3D np arrays) into a 4D np array
X = np.array(images)

# Flatten it, now each row represents a single image
dim1, dim2, chan = arr.shape
n_features = chan*dim1*dim2
X = X.reshape((len(images), n_features)) # flattened --> this goes to PCA 

# Init the model (a.k.a. specify the hyper-parameters e.g. number of components)
final_n_features = int(n_features*0.2)
pca = PCA(n_components=final_n_features)

# Transformed features
X_new = pca.fit_transform(X) # X_new has final_n_features --> this can be fed to the classfier model

# Define a classifer
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

