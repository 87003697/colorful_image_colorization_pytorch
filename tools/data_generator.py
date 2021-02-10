import numpy as np
import pdb
import cv2

from config import batch_size, img_rows, img_cols, nb_neighbors
import sklearn.neighbors as snn

from torch.utils.data import Dataset
import os
from PIL import Image

import torch
from torchvision import transforms
import torchvision

class Preprocessor():
    def __init__(self):
        # Load the array of quantized ab value
        q_ab = np.load("data/pts_in_hull.npy")
        self.nn_finder = snn.NearestNeighbors(n_neighbors=nb_neighbors, 
                                                     algorithm='ball_tree').fit(q_ab)
        self.nb_q = q_ab.shape[0]
        
    def _get_soft_encoding(self, image_ab, nn_finder, nb_q):
        
        h, w = image_ab.shape[:2]
        a = np.ravel(image_ab[:, :, 0])
        b = np.ravel(image_ab[:, :, 1])
        ab = np.vstack((a, b)).T
        # Get the distance to and the idx of the nearest neighbors
        dist_neighb, idx_neigh = nn_finder.kneighbors(ab)
        # Smooth the weights with a gaussian kernel
        sigma_neighbor = 5
        wts = np.exp(-dist_neighb ** 2 / (2 * sigma_neighbor ** 2))
        wts = wts / np.sum(wts, axis=1)[:, np.newaxis]
        # format the tar get
        y = np.zeros((ab.shape[0], nb_q))
        idx_pts = np.arange(ab.shape[0])[:, np.newaxis]
        y[idx_pts, idx_neigh] = wts
        y = y.reshape(h, w, nb_q)
        
        return y
    def __call__(self, data): # data: Image
        out_img_rows, out_img_cols = img_rows // 4, img_cols // 4
#         bgr = data.to('cpu').detach().numpy().transpose([1,2,0])
        bgr = np.asarray(data)
        gray = cv2.cvtColor(bgr, cv2. COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (img_rows, img_cols), cv2.INTER_CUBIC)
        x = gray / 255
        
        lab = cv2.cvtColor(bgr, cv2.COLOR_RGB2LAB)
        out_lab = cv2.resize(lab, (out_img_rows, out_img_cols), 
                            cv2.INTER_CUBIC)
        out_ab = out_lab[:, :, 1:].astype(np.int32) - 128
        y = self._get_soft_encoding(out_ab, self.nn_finder, self.nb_q)

        if np.random.random_sample() > 0.5:
            x = np.fliplr(x)
            y = np.fliplr(y)
            
        x_tensor = torch.tensor(x.copy(), dtype=torch.float32).unsqueeze(0)
        y_tensor = torch.tensor( y.transpose([2,0,1]).copy())
        return [x_tensor, y_tensor]
    def __repr__(self):
        return self.__class__.__name__+'()'
    
class ValDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, val_path,transform):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = val_path
        self.f_list = os.listdir(val_path)
        self.transform = transform

    def __len__(self):
        return len(self.f_list)

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, self.f_list[idx])
        im = Image.open(img_path).convert('RGB')
        if self.transform:
            x, gt = self.transform(im)        
        return x, gt