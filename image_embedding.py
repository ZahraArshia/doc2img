# =============================================================================
# libraries
# =============================================================================
import numpy as np
import argparse
import cv2 as cv
import pandas as pd

# =============================================================================
# decleration
# =============================================================================
filenn="demo.csv"
df = pd.read_csv('demo.csv', delimiter=',', header=0, usecols=[0,1])

# =============================================================================
# normalization
# =============================================================================
def scaling(series):
    minimum = np.amin(series)
    maximum = np.amax(series)
    new = np.zeros(len(series))
    for i in range(len(series)):
        new[i] = (series[i] - minimum)/(maximum - minimum)
    return new
	
# =============================================================================
# Binarization
# =============================================================================	
def binarization(matrix, threshold):
    matrix[matrix < threshold] = 0.0
    matrix[matrix >= threshold] = 1.0
    return matrix

# =============================================================================
# matrix to greyscale image
# =============================================================================	
def Mat2Image(matrix, fileName):
    minimun = np.amin(np.min(matrix))
    maximun = np.amax(np.amax(matrix))
    diff = maximun-minimun
    print("max= %.1f, min= %.1f"%(minimun,maximun))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i,j] = 255*((matrix[i,j]-minimun)/(diff))
    cv.imwrite(fileName, matrix)

# =============================================================================
# Recurrence Plot model : abs(series[x] - series[y])
# =============================================================================	 
def model(series, bin=0):
    dim = len(series)
    rp = np.zeros((dim,dim))
    for x in range(dim):
        for y in range(dim):
            rp[x,y] = abs(series[x] - series[y])
     if (bin == 1):
         rp = binarization(rp)
    return rp
# =============================================================================
# load data
# =============================================================================	 
def loadingData_x1(fileName):
        dataframe1 = pd.read_csv('demo.csv', delimiter=',', header=0, usecols=[0], engine='python') #first line is read as header
    return dataframe1
def loadingData_x2(fileName):
    dataframe2 = pd.read_csv(fileName, usecols=[1], engine='python') #first line is read as header
        return dataframe2

# =============================================================================
# create_series
# =============================================================================	 
def create_series(dataset,win_length, method):
    X = []
    Y = []
    length = len(dataset)
    
    if method == 'byword':
        for i in range(0, length, win_length):
             sequence = dataset[i:i + seq_length-1]
             label = dataset[i + win_length-1]
             X.append(sequence)
             Y.append(label)
    else: #byev
		ev = 0
        for i in range(0, length, win_length):
			 sequence = [] 
			 for j, k in zip(range(0,length/win_length -1,1), range(ev,length-(win_lenght-1)-ev,win_lenght):
             sequence[j] = dataset[k]
		 label = dataset[i + win_length-1]
		 X.append(sequence)
		 Y.append(label)
		 ev = ev+1
    return X,Y

# =============================================================================
# series to image
# =============================================================================	 
def series2image(win_lenght)
	dimension = win_lenth
	half_len_win = int(win_lenth / 2)
	win = []
	windows = []
    print('win_length', win_lenth)
    number = 0
    # for i in range(half_len, len(x), half_len):
    for i in range(half_len_win, len(df), half_len_win):
        win = []

        counter = i - half_len_win
        while (counter < dimension):
            dataframe1 = loadingData_x1(filenn)
            win1 = dataframe1.iloc[counter:dimension]
            win1 = win1.reset_index().values.ravel()
            win.append(win1)
            win1 = win1[0::1]
			print('==============')
           
            dataframe2 = loadingData_x2(filenn)
            win2 = dataframe2.iloc[counter:dimension]
            win2 = win2.reset_index().values.ravel()
            win.append(win2)
            win2 = win2[0::1]
            
            windows.extend(win1)
            windows.extend(win2)
            counter = counter + 1
            dimension = dimension + 1
            number = number + 1

           
            print('dimension')
            print(dimension)
            
            if dimension >= len(dataset-win_length): #len(dataset-win_length)
                break;

            
            dataset = windows
            windows = []
        
            print(dataset)
            # y_label = df[]
            y_label = dataframe1.iloc[kj]
            label = y_label.reset_index().values.ravel()
            ylabel = label[1]

            print("label_y: "+str(ylabel))
            new_dataset = scaling(dataset)
            rp = []
            rp = model(new_dataset, bin=0)
            first = np.amin(dataset[1]) #smallest value in dataset 
            for index in range(np.amin(dataset[1]), np.amax(dataset[1]), win_lenght)): 
											
               
                if index <= ylabel < (index+win_lenth): #win_lenght
                    yclass = "c"+str(index)
                    print("class:" + str(yclass))
                    path = "data/"+str(yclass)

                    try:
                        os.mkdir(path)
                    except OSError:
                        print("Creation of the directory %s failed" % path)
                    else:
                        print("Successfully created the directory %s " % path)
                    Mat2Image(rp, "data/"+str(yclass)+"/RP"+str(number)+","+ str(ylabel)+","+str(yclass) + ".jpg")
    print('finish win creation')


# =============================================================================
# main
# =============================================================================	 
if __name__ == "__main__":
    import os
    win_lenth = 15
	series2image(win_lenght)
    
    
    

