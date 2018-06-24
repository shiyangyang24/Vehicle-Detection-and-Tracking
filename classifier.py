import glob
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import hog
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import time
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
SEED = 2018

# Read dataset image
vehicle_images = glob.glob('vehicles/GTI*/*.png')
none_vehicle_images = glob.glob('non-vehicles/*/*.png')
cars = []
notcars = []
for image in vehicle_images:
    cars.append(image)
for image in none_vehicle_images:
    notcars.append(image)
print('Dataset size:Cars {} | NotCars {}'.format(len(cars),len(notcars)))

#接下来，我们来提取Histogram of oriented Gradients(HOG)特征。
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm= 'L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       block_norm= 'L2-Hys',
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs,color_space="RGB",spatial_size=(32,32),
                    hist_bins=32,orient=9,
                    pix_per_cell=8,cell_per_block=2,hog_channel=0,
                    spatial_feat=True,hist_feat=True,hog_feat=True,
                    hog_vis=False):
    '''
    Feature extractor:extract features from a list of images
    The function calls bin_spatial(),color_hist() and get_hog_features
    '''
    #create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        if hog_vis == False:
            image = image.astype(np.float32)/255
        # apply color conversion if other than 'RGB'
        # color conversion
        if color_space in ['HSV','LUV','HLS','YUV','YCrCb']:
            feature_image = cv2.cvtColor(image,eval('cv2.COLOR_RGB2'+color_space))
        else: feature_image = np.copy(image)
        # Image size: add all pixels of reduced image as vector
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image,size=spatial_size)
            #print('spatial features shape:',spatial_features.shape)
            file_features.append(spatial_features)
        # Histogram of reduced image: add histogram as a vector
        if hist_feat == True:
            hist_features = color_hist(feature_image,nbins=hist_bins)
            file_features.append(hist_features)
        #HOG of reduced image: add HOG as feature vector
        if hog_feat == True:# Call get_hog_features() with vis=False ,feature_vec = True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    if hog_vis:
                        hog_feature,hog_image = get_hog_features(feature_image[:,:,channel],
                                                                orient,pix_per_cell,cell_per_block,
                                                                vis=True,feature_vec=True)
                        #print(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).dtype)
                        res = cv2.addWeighted(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),0.1,
                                              ((hog_image/np.max(hog_image))*255).astype(np.float32),0.1,0.0)
                        # Plot the examples
                        fig = plt.figure()
                        plt.title(channel)
                        plt.subplot(131)
                        plt.imshow(image,cmap='gray')
                        plt.title('Original Image')
                        plt.subplot(132)
                        plt.imshow(hog_image,cmap='gray')
                        plt.title('HOG')
                        plt.subplot(133)
                        plt.imshow(res,cmap='gray')
                        plt.title('overlapped')
                        plt.show()
                    else:
                        hog_feature = get_hog_features(feature_image[:,:,channel],
                                                      orient,pix_per_cell,cell_per_block,
                                                      vis=False,feature_vec=True)
                    #print('hog feature shape:',hog_feature.shape)
                    hog_features.append(hog_feature)
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel],orient,
                                               pix_per_cell,cell_per_block,vis=False,feature_vec = True)
            #Append the new feature vector to the features list
            #print('hog features shape:',hog_features.shape)
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
        #print(np.concatenate(file_features).shape)
    # return list of feature vectors
    return features


color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
hist_range = bins_range = (0,256)
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


rand_img = np.random.choice(np.arange(0,len(notcars),1))

print('Image adress:',notcars[rand_img])
feat = extract_features([notcars[rand_img]],color_space=color_space,
                        spatial_size=spatial_size,hist_bins=hist_bins,
                        orient=orient,pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel,spatial_feat=spatial_feat,
                        hist_feat=hist_feat,hog_feat=hog_feat,hog_vis=True
                       )



rand_img = np.random.choice(np.arange(0,len(cars),1))

print('Image adress:',cars[rand_img])
feat = extract_features([cars[rand_img]],color_space=color_space,
                        spatial_size=spatial_size,hist_bins=hist_bins,
                        orient=orient,pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel,spatial_feat=spatial_feat,
                        hist_feat=hist_feat,hog_feat=hog_feat,hog_vis=True
                       )










#Run linear SVC classifier
car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)
    
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()




#Logistic Regression Classifier


lrc = LogisticRegression(max_iter=10)
t = time.time()
lrc.fit(X_train,y_train)
t2 = time.time()
print(round(t2-t,2),'Seconds to train LRC...')
# Check the score of the LRC
print('Train Accuracy of LRC=',round(lrc.score(X_train,y_train),4))
print('Test Accuracy of LRC=',round(lrc.score(X_test,y_test),4))
# Check the prediction time for a single sample
t = time.time()
n_predict = 10
print('My LRC predicts:',lrc.predict(X_test[0:n_predict]))
print('For these',n_predict,'labels:',y_test[0:n_predict])
t2 = time.time()
print(round(t2-t,5),'Seconds to predict',n_predict,'labels with LRC')





mlp = MLPClassifier(random_state=SEED)
t = time.time()
mlp.fit(X_train,y_train)
t2 = time.time()
print(round(t2-t,2),'Seconds to train MLP...')
# Check the score of the LRC
print('Train Accuracy of MLP=',round(mlp.score(X_train,y_train),4))
print('Test Accuracy of MLP=',round(mlp.score(X_test,y_test),4))
# Check the prediction time for a single sample
t = time.time()
n_predict = 10
print('My MLP predicts:',mlp.predict(X_test[0:n_predict]))
print('For these',n_predict,'labels:',y_test[0:n_predict])
t2 = time.time()
print(round(t2-t,5),'Seconds to predict',n_predict,'labels with LRC')



model_combine = 'car_detection.p'
try:
    with open(model_combine,'wb') as pfile:
        pickle.dump(
        {
            'X_dataset':X,
            'y_dataset':y,
            'svc':svc,
            'lrc':lrc,
            'mlp':mlp,
            'X_scaler':X_scaler,
            'color_space':color_space,
            'spatial_size':spatial_size,
            'hist_bins':hist_bins,
            'orient':orient,
            'pix_per_cell':pix_per_cell,
            'cell_per_block':cell_per_block,
            'hog_channel':hog_channel,
            'spatial_feat':spatial_feat,
            'hist_feat':hist_feat,
            'hog_feat':hog_feat
        },
            pfile,pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to',car_detection,':',e)
    raise
