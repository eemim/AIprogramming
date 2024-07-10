# test harness for evaluating models on the cifar10 dataset
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD


# plot diagnostic learning curves
def plot_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.legend()
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.show()  

def init_model():
    global model
    # load data
    X=np.load('trainx.npy')
    Y=np.load('trainy.npy')
    
    N_test=2000
    testX=X[:N_test, :, :]  
    testY=Y[:N_test, :]  
    
    trainX=X[N_test:, :, :]  
    trainY=Y[N_test:, :]  
    
    del X, Y
      
    model = Sequential()
    
    #extra parameter padding='same' forces output of the convolution to be
    #same size as original image. Use padding='same' with all convolutional
    #layers in this model.

    #Your code starts here.
    #Read https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/ 
    #Use "The updated VGG 3"-model
    

    model.add(Conv2D(32, kernel_size=3, input_shape=(32,32,1), kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(32, kernel_size=3, kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, kernel_size=3, kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, kernel_size=3, kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, kernel_size=3, kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, kernel_size=3, kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    #more rows here
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    #your code ends here
    
    model.summary()
    
    # VAIHDETTU lr => learning_rate
    opt = SGD(learning_rate=0.001, momentum=0.9)  
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(trainX, trainY, epochs=60, batch_size=64, validation_data=(testX, testY), verbose=1)

    model.save('my_code.h5')

    return testX, testY, history

def load_my_model(filename='my_code.h5'):
    global model
    model=load_model(filename)

def compute_values(x): #x is vector of images
    global model
    return model.predict(x)

if __name__ == "__main__":
    x_test, y_test, history=init_model()

    #_, acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred=compute_values(x_test)
    y_pred_idx=np.argmax(y_pred, axis=1)
    y_test_idx=np.argmax(y_test, axis=1)
    
    N_correct=(y_pred_idx==y_test_idx).sum()
    N_all=np.shape(y_test)[0]
    acc=N_correct/N_all
    print('Accuracy: %.3f'%(acc))
    plot_diagnostics(history)

