# flowers-recognition
Assignment of AICA  
### Args:
lr--learning rate  
batch_size--training batch size  
epochs--training epochs  
reg--regularizer rate  
method--'vgg','svm','knn','fc','resnet34','resnet50'  
### Run:
eg. to run vgg, the step is:  
cd flowers-recognition  
python recognition.py --method vgg --lr 0.0001 --epochs 10000 --reg 0.005 --batch_size 64  
### Env:
python 3.x  
tensorflow 1.11.0 cuda9.0  
numpy cv2 etc.  
### Data:
the flowers dataset can be found in https://www.kaggle.com/alxmamaev/flowers-recognition/home  
the pre-trained ResNet-50 can be found in https://www.kaggle.com/cokastefan/keras-resnet-50/data  
### Details：
There are some dirfferences from original model(VGG,ResNet).  
Using dropout to prevent over-fitting  
Using 1*1 convolution layer to perform dimension reduction  
Using batch normalization to help training  
Adding regularizer to prevent overfitting  
For KNN and SVM,  
Using validation set to select the best k, learning rate and regularizer rate.  
