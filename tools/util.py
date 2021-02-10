from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed

import numpy as np
import cv2 as cv
from config import img_rows, img_cols
from config import nb_neighbors, T, epsilon
import sklearn.neighbors as snn
import torch

def load_img(img_path):
    out_np = np.asarray(Image.open(img_path))
    if(out_np.ndim==2):
        out_np = np.tile(out_np[:,:,None],3)
    return out_np

def resize_img(img, HW=(256,256), resample=3):
    return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
    # return original size L and resized L as torch Tensors
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_orig = img_lab_orig[:,:,0]
    img_l_rs = img_lab_rs[:,:,0]

    tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
    tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]

    return (tens_orig_l, tens_rs_l)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
    # tens_orig_l 	1 x 1 x H_orig x W_orig
    # out_ab 		1 x 2 x H x W

    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    # call resize function if needed
    if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
    else:
        out_ab_orig = out_ab

    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))


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
