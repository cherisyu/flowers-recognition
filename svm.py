'''using support vector machine'''
import numpy as np
import load_data as ld
import random
import time




def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs and outputs are the same as svm_loss_naive.
  """
 # loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1) #(N, 1)
  margins = np.maximum(0, scores - correct_class_scores +1)
  margins[range(num_train), list(y)] = 0
  loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)

  coeff_mat = np.zeros((num_train, num_classes))
  coeff_mat[margins > 0] = 1
  coeff_mat[range(num_train), list(y)] = 0
  coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

  dW = (X.T).dot(coeff_mat)
  dW = dW/num_train + reg*W
  return loss, dW

def train(X, y, learning_rate, reg, num_iters=200, batch_size=200):
  """
  Train this linear classifier using stochastic gradient descent.
  Inputs:
  - X: A numpy array of shape (N, D) containing training data; there are N
    training samples each of dimension D.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c
     means that X[i] has label 0 <= c < C for C classes.
  - learning_rate: (float) learning rate for optimization.
  - reg: (float) regularization strength.
  - num_iters: (integer) number of steps to take when optimizing
  - batch_size: (integer) number of training examples to use at each step.
  - verbose: (boolean) If true, print progress during optimization.
  Outputs:
  A list containing the value of the loss function at each training iteration.
  """
  num_train=X.shape[0]
  dim = X.shape[1]
  num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
 # if W is None:
    # initialize W
  W = 0.001 * np.random.randn(dim, num_classes) # returns a set of samples with standard normal distributions

  # Run stochastic gradient descent to optimize W
  #loss_history = []
  loss=0
  for it in range(num_iters):
    X_batch = None
    y_batch = None
    #The generated random numbers can be duplicated
    batch_idx = np.random.choice(num_train, batch_size, replace = True)
    X_batch = X[batch_idx]
    y_batch = y[batch_idx]

    # evaluate loss and gradient
    loss, grad = svm_loss_vectorized(W, X_batch, y_batch, reg)
    #loss_history.append(loss)

    # Update the weights using the gradient and the learning rate.
    W += - learning_rate * grad
  print (' iteration %d / %d: loss %f ' % (it+1, num_iters, loss))

  return W

def predict(X):
  """
  Use the trained weights of this linear classifier to predict labels for
  data points.
  Inputs:
  - X: D x N array of training data. Each column is a D-dimensional point.
  Returns:
  - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
    array of length N, and each element is an integer giving the predicted
    class.
  """
  y_pred = np.zeros(X.shape[0])
  scores = X.dot(W)
  y_pred = np.argmax(scores, axis = 1)
  return y_pred

def accuracy(y,y_pred):
  num=len(y_pred)
  sum=0
  for i in range(num):
    if y[i]==y_pred[i]:
      sum+=1
  acc=sum/num
  return acc


def run_svm():

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

    learning_rates = [1.4e-4, 1.5e-4, 1.6e-4]
    regularization_strengths = [(1+i*0.1)*1e-4 for i in range(-3,3)] + [(2+0.1*i)*1e-4 for i in range(-3,3)]

    all_W=[]
    all_acc=[]
    for i in range(len(learning_rates)):
      for j in range(len(regularization_strengths)):
        W=train(Train_img_flatten, Train_label, learning_rates[i],regularization_strengths[j], num_iters=200, batch_size=200)
        all_W.append(W)
        predict_label=predict(Validation_img_flatten)
        acc=accuracy(Validation_label,predict_label)
        all_acc.append(acc)
        print("learning_rates=%f,regularization_strengths=%f,accuracy=%f"%(learning_rates[i],regularization_strengths[j],acc))

    index=np.argmax(all_acc)
    W=all_W[index]
    i=index/len(regularization_strengths)
    j=index%len(regularization_strengths)

    Test_predict=predict(Test_img_flatten)
    acc=accuracy(Test_label,Test_predict)
    print("The accuracy of test set is %f"%acc)

