

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import random 
import numpy as np
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
from skimage.filters import gabor_kernel , gabor
########################################
#c)
def data():
  ds=fetch_openml('mnist_784',as_frame=False)
  x,x_test,y,y_test = train_test_split(ds.data,ds.target,test_size=0.2,random_state=42)

  #getting 10000 random indexes
  random_idx=random.sample(range(x.shape[0]),10000)

  x_new=x[random_idx]
  y_new = y[random_idx]
  
  #reshaping vector of length 784 to a matrix of size 28x28 for downsampling
  x_new=x_new.reshape((-1,28,28))
  x_test=x_test.reshape((-1,28,28))
  x_test_old=x_test
  #initializing a new down sample matrix
  d_image = np.zeros((x_new.shape[0],14*14))

  #resizing the 28x28 matrix to 14x14 matrix and flattening it
  for i,val in enumerate(x_new):
    d_image[i] = cv2.resize(x_new[i],(14,14)).flatten()

  #Splitting the data into train and validation sets

  x_train,x_val,y_train,y_val=train_test_split(d_image,y_new,test_size=0.2,random_state=42)
  print(x_train.shape)
  print(y_train.shape)
  #down sampling the test images
  d_image_test = np.zeros((x_test.shape[0],14*14))
  for i,val in enumerate(x_test):
    d_image_test[i] = cv2.resize(x_test[i],(14,14)).flatten()

  x_test=d_image_test
  
  return x_train,x_val,y_train,y_val,x_test,y_test,x_new,x_test_old,d_image,y_new

def show_images(x_train,x_train_old,x_test,x_test_old,y_train):
  #Visualizing the images before and after down-sampling
  fig, axs = plt.subplots(1,2)
  axs[0].imshow(x_train_old[2].reshape(28,28))
  axs[1].imshow(x_train[2].reshape(14,14))
  axs[0].set_title('Training images original')
  axs[1].set_title('Training images down-sampled')
  axs[0].set_xlabel(f'label = {y_train[2]}')
  axs[1].set_xlabel(f'label = {y_train[2]}')

  fig1, axs1 = plt.subplots(1,2)
  axs1[0].imshow(x_test_old[2].reshape(28,28))
  axs1[1].imshow(x_test[2].reshape(14,14))
  axs1[0].set_title('Testing images original')
  axs1[1].set_title('Training images down-sampled')
  axs1[0].set_xlabel(f'label = {y_test[2]}')
  axs1[1].set_xlabel(f'label ={y_test[2]}')

#######################################
#d)

def train(x_train,y_train):
  clf = make_pipeline(StandardScaler(), svm.SVC(C=10,kernel='poly',gamma='scale'))
  clf.fit(x_train,y_train)
  return clf

def confusion_matrix(x_test,y_test):
  matrix=plot_confusion_matrix(clf,x_test,y_test,cmap=plt.cm.Reds)
  matrix.ax_.set_title('Confusion matrix',color='white')
  plt.show()

##################################
#g)
def grid(x_train,y_train):
  
  clf = GridSearchCV(svm.SVC(), param_grid=[{'C':[10,100,1000], 'gamma':[1e-4,1e-5,1e-6], 'kernel':['poly','rbf']}], scoring='accuracy')
  clf.fit(x_train[:3000], y_train[:3000])
  return clf

def best_estimator(clf1):
  df=pd.DataFrame(clf1.cv_results_)[['params','mean_test_score']]
  print(df)
  classifier = clf1.best_estimator_
  print('Val accuracy = ',classifier.score(x_val,y_val))
  print('Best parameters = ',clf1.best_params_)

#########################################
#i)
def plot_gabor():
  for theta in np.arange(0,np.pi,np.pi/4):
      for freq in np.arange(0.05,0.5,0.15):
        for bandwidth in np.arange(0.3,1,0.3):
              gk = gabor_kernel(frequency=freq, theta=theta, bandwidth=bandwidth)
              f, axarr = plt.subplots(1,2)
              axarr[0].imshow(gk.real)
              axarr[1].imshow(gk.imag)

def gabor_data_setup(x_train,y_train,x_val,y_val): 
  xx_train=np.zeros([1000,196])
  yy_train=np.zeros([1000,])
  z=0
  for i in range(0,10):
    lst=np.where(y_train==f'{i}')
    random_idx=random.sample(list(lst[0]),100)
    xx_train[z:100+z]=x_train[random_idx]
    yy_train[z:100+z]=y_train[random_idx]
    z+=100


  xx_val=np.zeros([1000,196])
  yy_val=np.zeros([1000,])
  z=0
  for i in range(0,10):
    lst=np.where(y_val==f'{i}')
    random_idx=random.sample(list(lst[0]),100)
    xx_val[z:100+z]=x_val[random_idx]
    yy_val[z:100+z]=y_val[random_idx]
    z+=100
  return xx_train,yy_train,xx_val,yy_val


def gabor_SVM(xx_train,yy_train,xx_val,yy_val):
  xx_new_train=np.zeros([1000,14*14*36])
  xx_new_val=np.zeros([1000,14*14*36])
  for a in range(1000):
    l=0
    for theta in np.arange(0,np.pi,np.pi/4):
      for freq in np.arange(0.05,0.5,0.15):
        for bandwidth in np.arange(0.3,1,0.3):
          gk = gabor_kernel(frequency=freq, theta=theta, bandwidth=bandwidth)
          image = xx_train[a].reshape((14,14))
          image_val = xx_val[a].reshape((14,14))
          coeff_real_tr , _ = gabor(image, frequency=freq, theta=theta,bandwidth=bandwidth)
          coeff_real_te , _ = gabor(image, frequency=freq, theta=theta,bandwidth=bandwidth)
          xx_new_train[a,l:l+196]=coeff_real_tr.flatten()
          xx_new_val[a,l:l+196]=coeff_real_te.flatten()
          l+=196
  clf = svm.SVC(C=1.0, kernel='poly', gamma='auto', cache_size=1000, verbose=True)
  clf.fit(xx_train,yy_train)

  print(clf.score(xx_val, yy_val))


x_train,x_val,y_train,y_val,x_test,y_test,x_train_old,x_test_old,d_image,yy=data()
show_images(d_image,x_train_old,x_test,x_test_old,yy)
clf=train(x_train,y_train)
confusion_matrix(x_test,y_test)
clf1=grid(x_train,y_train)
best_estimator(clf1)
plot_gabor()
xx_train,yy_train,xx_val,yy_val=gabor_data_setup(x_train,y_train,x_val,y_val)
# gabor_SVM(xx_train,yy_train,xx_val,yy_val)  #Uncomment this to run gabor SVM