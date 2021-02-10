import numpy as np
import cv2 as cv
from config import img_rows, img_cols
from config import nb_neighbors, T, epsilon
import sklearn.neighbors as snn
import torch

def display(my_model, filename):
    h, w = img_rows // 4, img_cols // 4

    # Load the array of quantized ab value
    q_ab = np.load("data/pts_in_hull.npy")
    nb_q = q_ab.shape[0]

    # Fit a NN to q_ab
    nn_finder = snn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)
    
    bgr_raw = cv.imread(filename)
    gray_raw = cv.imread(filename, 0)
    orig_shape = bgr_raw.shape[:2]
    orig_h = bgr_raw.shape[1]
    orig_w = bgr_raw.shape[0]
    
    bgr = cv.resize(bgr_raw, (img_rows, img_cols), cv.INTER_CUBIC)
    gray = cv.resize(gray_raw, (img_rows, img_cols), cv.INTER_CUBIC)
    # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    
    x_test = np.empty((1, 1,img_rows, img_cols), dtype=np.float32)
    x_test[0, 0, :, :] = gray / 255.

    # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
    X_colorized,_ = my_model(torch.tensor(x_test))
    X_colorized = X_colorized.cpu().detach().numpy().reshape((h * w, nb_q))

    # Reweight probas
    X_colorized = np.exp(np.log(X_colorized + epsilon) / T)
    X_colorized = X_colorized / np.sum(X_colorized, 1)[:, np.newaxis]

    # Reweighted
    q_a = q_ab[:, 0].reshape((1, 313))
    q_b = q_ab[:, 1].reshape((1, 313))

    X_a = np.sum(X_colorized * q_a, 1).reshape((h, w))
    X_b = np.sum(X_colorized * q_b, 1).reshape((h, w))
    
    X_a = cv.resize(X_a, (img_rows, img_cols), cv.INTER_CUBIC)
    X_b = cv.resize(X_b, (img_rows, img_cols), cv.INTER_CUBIC)
    X_a = X_a + 128
    X_b = X_b + 128
    out_lab = np.zeros((img_rows, img_cols, 3), dtype=np.int32)
    out_lab[:, :, 0] = lab[:, :, 0]
    out_lab[:, :, 1] = X_a
    out_lab[:, :, 2] = X_b
    out_L = out_lab[:, :, 0]
    out_a = out_lab[:, :, 1]
    out_b = out_lab[:, :, 2]
    out_lab = out_lab.astype(np.uint8)
    out_rgb = cv.cvtColor(out_lab, cv.COLOR_LAB2RGB)
    # print('np.max(out_bgr): ' + str(np.max(out_bgr)))
    # print('np.min(out_bgr): ' + str(np.min(out_bgr)))
    out_rgb = out_rgb.astype(np.uint8)

    out_rgb = cv.resize(out_rgb,(orig_h, orig_w), cv.INTER_CUBIC)
    rgb_raw = cv.cvtColor(bgr_raw, cv.COLOR_BGR2RGB)
    gray_raw = cv.cvtColor(bgr_raw, cv.COLOR_BGR2GRAY)
    
    return rgb_raw, gray_raw,  out_rgb