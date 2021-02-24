# =============================================================================
# libraries
# =============================================================================
import numpy as np
import pandas as pd
import math
from keras.preprocessing.image import  img_to_array
from skimage import transform
from deep_ranking import deep_rank_model
import skimage
import matplotlib.pyplot as plt

# =============================================================================
# matlab code to python(I don't actually know what the hell this code do!)
# =============================================================================
def MyfGetRcImgM1(s, d, tau):

    y = np.transpose(s)
    N = len(y)
    N2 = N - tau * (d - 1)

    xe = []
    for mi in range(d):
        y1 = y[0:N2]
        te = []
        for i in range(len(y1)):
            te.append(y1[i] + tau * mi)
        xe.append(te)
    
    xe = np.transpose(xe) 
    x1 = np.tile(xe, [N2, 1])    
    xe = np.transpose(xe)    
    m1 = np.reshape(xe,(N2*d , 1))
    mm = np.tile(m1, (1 , N2));
    x2 = np.reshape(mm,(d,N2*N2))
    x2 = np.transpose(x2)    
    f1 = x1-x2
    
    f2 = []
    for i in range(d):
        ff = f1[:,i]
        s1 = []
        for j in ff:
            s1.append(math.pow(j,2))
        f2.append(s1)    
    f2 = np.transpose(f2)
    
    S = []
    for row in f2:
        S.append(math.sqrt(sum(row)))
    S = np.reshape(S, (N2, N2))
    
    return np.array(S)



# =============================================================================
#  create series and labeling(the next word position is label )
# =============================================================================
def create_series(dataset,seq_length, method):
    X = []
    Y = []
    length = len(dataset)
    
    if method == 'by ev': #errrrrrrrrrrrrrror
		sequence = np.zeros(length/seq_length)
        for i in range(0, length-seq_length, 1):
             temp_sequence = dataset[i:i + seq_length]
			 sequence[i] = temp_sequence[i]
             label = dataset[i + seq_length] #errrrrrrrrrrrrrror
             X.append( sequence)
             Y.append(label)
    else:
        for i in range(0, length, seq_length):
             sequence = dataset[i:i + seq_length-1]
             label = dataset[i + seq_length-1]
             X.append( sequence)
             Y.append(label)
      
    return X,Y



# =============================================================================
#  load data
# =============================================================================
def load_data(look_back , method, range_normalize):
    # load the dataset
    dataframe = []
    dataframe = pd.read_csv('data.csv')
    dt = dataframe.values
	GrundTruth = dt[:,0] #first column of data(posion)
    value = dt[:,1] #second column of data(values)
    #Normalize
    minimum = np.amin(value)
    maximum = np.amax(value)
    
    d = range_normalize[0]
    b = range_normalize[1]
    m = (b - d) / (maximum - minimum)
    value = (m * (value - minimum)) + d

    X, Y = create_series(value, look_back, method)
   
    return X, Y

# =============================================================================
#  embedding images (based on that stupid matlab code!)
# =============================================================================
def create_image_embed(X, model, nTauShiftAmnt, nDimNumOfShifts):
    
    print('Embedding images...')
    embedding = []
    all_Image = [] #images themselves
    
    for i in range(len(X)):
        if i%10 == 0 and len(X) > 100:
            print('image: \t' + str(i+1) + '\t from \t' + str(len(X)))
        adInpAr = X[i]
        Irp1 = MyfGetRcImgM1(adInpAr, nTauShiftAmnt, nDimNumOfShifts)  #grey scal image
        Irp2 = skimage.color.gray2rgb(Irp1) 							#RGB image
        all_Image.append(Irp2)
        Irp = Irp2            
        Irp = img_to_array(Irp).astype("float64")
        Irp = transform.resize(Irp, (224, 224))
        Irp *= 1. / 255
        Irp = np.expand_dims(Irp, axis = 0)
        embedding.append(model.predict([Irp,Irp,Irp])[0])
    return all_Image, embedding

# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    
    nTauShiftAmnt = 4
    nDimNumOfShifts = 3
    num_of_class = 5
    range_normalize = [1,10]
    
    look_back = 300 #deries window
    method = 'by ev' # 'by word' , 'by ev'
    
    model = deep_rank_model()
    
    X, Y = load_data(look_back, method, range_normalize)
    
    all_Image, embedding = create_image_embed(X, model, nTauShiftAmnt, nDimNumOfShifts)


#    np.save('Embed5000_seprate.npy', embedding)
#    embedding = np.load('Embed200.npy') #  Embed5000_seprate, Embed200
    
    
    label, set_label, head_img, all_head = get_label(X, Y, model, num_of_class, train_range_split, method)
    
    
    
    