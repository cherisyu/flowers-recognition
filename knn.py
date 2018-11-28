'''using k-nearest neighbor'''
import numpy as np
import load_data as ld
import random
import time



def compute_distances(Test_img,Train_img):
    """
    Compute the distance between each test point in Test_img and each training point
    in Train_img using a nested loop over both the training data and the test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training point.
    """
    num_test = Test_img.shape[0]
    num_train = Train_img.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        dists[i][j] = np.sqrt(np.sum(np.square(Test_img[i]-Train_img[j])))
    return dists

def predict_labels(dists, k, Train_label):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      closest_y = []
      closest_y = Train_label[np.argsort(dists[i,:])[:k]]
      y_pred[i]=np.argmax(np.bincount(closest_y))
    return y_pred

def accuracy(Test_label,y_pred):
    num=len(y_pred)
    sum=0
    for i in range(num):
        if Test_label[i]==y_pred[i]:
            sum+=1
    acc=sum/num
    return acc

def run_knn():

    F = ld.Flower()
    flowers,Image,Label,Label_onehot = F.read_img()
    Train_img,Train_label,Validation_img,Validation_label,Test_img,Test_label = F.split_data(flowers,Image,Label,Label_onehot,returnwhat=1)

    shape=np.shape(Train_img)
    N=shape[0]
    dim=shape[1]*shape[2]*shape[3]
    Train_img_flatten=np.reshape(Train_img,(N,dim))

    shape=np.shape(Validation_img)
    N=shape[0]
    dim=shape[1]*shape[2]*shape[3]
    Validation_img_flatten=np.reshape(Validation_img,(N,dim))

    shape=np.shape(Test_img)
    N=shape[0]
    dim=shape[1]*shape[2]*shape[3]
    Test_img_flatten=np.reshape(Test_img,(N,dim))

    all_acc=[]
    dists= compute_distances(Validation_img_flatten,Train_img_flatten)
    for k in range(1,11):
        validation_pred=predict_labels(dists, k, Train_label)
        acc=accuracy(Validation_label,validation_pred)
        all_acc.append(acc)
        print("k = %d , accuracy is %f" % (k,acc))

    index=np.argmax(all_acc)
    k=index+1
    dists= compute_distances(Test_img_flatten,Train_img_flatten)
    test_predict=predict_labels(dists, k, Train_label)
    acc=accuracy(Test_label,test_predict)
    print("k = %d , The accuracy of test set is %f" % (k , acc))


