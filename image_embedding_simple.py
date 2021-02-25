# =============================================================================
# libraries
# =============================================================================
import numpy as np
import argparse
import cv2 as cv
import pandas as pd
import os


# =============================================================================
# normalization
# =============================================================================
def scaling(series):
    minimum = np.amin(series)
    maximum = np.amax(series)
    new = np.zeros(len(series))
    dist = maximum - minimum
    for i in range(0, len(series)):
        new[i] = (series[i] - minimum) / dist
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
    maximum = np.amax(np.amax(matrix))
    diff = maximum - minimun
    print("max= %.1f, min= %.1f" % (minimun, maximum))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = 255 * ((matrix[i, j] - minimun) / (diff))
    cv.imwrite(fileName, matrix)


# =============================================================================
# Recurrence Plot model : abs(series[x] - series[y])
# =============================================================================	 
def model(X):
    Xp = []
    for series in X
        dim = len(series)
        i_rp = np.zeros(dim)
        for x in range(dim):
            i_rp[x] = abs(255*series[x])
        Xp.append(i_rp)
    return Xp


# =============================================================================
# load data
# =============================================================================	 
def loading_data_x1(file_name):  # fileName:'demo.csv'
    data_frame_1 = pd.read_csv(file_name, delimiter=' ', header=0, usecols=[0],
                               engine='python')  # first line is read as header
    return data_frame_1


def loading_data_x2(file_name):
    data_frame_2 = pd.read_csv(file_name, delimiter=' ', header=0, usecols=[1],
                               engine='python')  # second line is read as header
    return data_frame_2


# =============================================================================
# create_series
# =============================================================================	 
def create_series(i_dataset, i_win_length):
    X = []
    length = len(i_dataset)
    ev = 0

    for i in range(0, i_win_length):
        sequence = []
        for j in range(0, length, i_win_length):
            k = j + i
            v = i_dataset[k]
            sequence.append(v)

        # label = i_dataset[i + i_win_length - 1]
        X.append(sequence)
        # Y.append(label)
        ev = ev + 1
    return X


# for j, k in zip(range(0, int(length / i_win_length) - 1, 1),
#                 range(ev, length - (i_win_length - 1) - ev, i_win_length)):
#     sequence[j] = i_dataset[k]


# =============================================================================
# series to image
# =============================================================================	 
def series2image(rp, win_length, path):

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


    for series in rp:
        dim = len(series)
        i_rp = np.zeros(dim)
        for x in range(dim):
            i_rp[x] = abs(255*series[x])
    number = 0
    df1 = dataset.iloc[:, 1]
    first = np.amin(df1)  # smallest value in dataset
    for index in range(np.amin(dataset[1]), np.amax(dataset[1]), win_length):
        number = number + 1
    Mat2Image(rp, os.path.join(path, "rp", str(number) + ".jpg"))
    print('finish win creation')


# =============================================================================
# main
# =============================================================================	 
if __name__ == "__main__":

    win_length = 300
    filename = "data/sample_final_csv"
    dataset = pd.read_csv(filename, delimiter=' ', header=None)
    value_col = dataset.iloc[:, 1]
    new_dataset = scaling(value_col)
    X = create_series(new_dataset, win_length)
    rp = model(X)
    series2image(rp, win_length, "data")
