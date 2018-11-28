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
