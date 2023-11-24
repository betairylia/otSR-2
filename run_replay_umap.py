import os
import numpy as np
from osrparse import Replay, KeyTaiko
import multiprocessing
import tqdm
import umap
import sklearn
import pickle
import scipy
import matplotlib, random
import pandas as pd
from matplotlib import pyplot as plt

from PIL import Image
from matplotlib import cm
import base64
from io import BytesIO

BPM = 138
slowThreshold = 60000 / (BPM * 4)
    
def swapLdk(arr):
    arr[[0, 1]] = arr[[1, 0]]
    arr[:, [0, 1]] = arr[:, [1, 0]]
    return arr

dds = np.array([[1,1], [2,1], [1,2], [2,2]])
dks = np.array([[1,0], [2,0], [1,3], [2,3]])
kks = np.array([[0,0], [0,3], [3,0], [3,3]])
kds = np.array([[0,1], [0,2], [3,1], [3,2]])
entries = np.stack([dds, dks, kks, kds], axis = 0)

def take_2d_entries(arr, ix):
    return arr[ix[..., 0], ix[..., 1]]

def normKD(arr):
    sums = take_2d_entries(arr, entries).sum(axis = -1, keepdims = True)

    if np.sum(sums) == 0:
        return arr
    
    min_non_zero = np.min(sums[sums > 0])
    sums[sums == 0] = min_non_zero
    
    arr = np.copy(arr)
    
#     print(arr)
    arr[entries[..., 0], entries[..., 1]] /= sums

#     print("Normalized:")
#     print(arr)
    
    return arr

def get_features(rep, slowThreshold):
    
    pressed = KeyTaiko(0)
    timestamp = 0 # Time since previous hit
    prevHit = KeyTaiko(0)

    slow_mat = np.zeros((4,4))
    fast_mat = np.zeros((4,4))

    for hit in rep.replay_data:
        newKeys = hit.keys & ~pressed
        pressed = hit.keys
        timestamp += hit.time_delta

        # New hits
        if newKeys != 0:

            # Assign to the array
            prev = np.array([i for i in range(4) if (((1 << i) & prevHit) > 0)], np.int8)
            curr = np.array([i for i in range(4) if (((1 << i) & newKeys) > 0)], np.int8)
            xx, yy = np.meshgrid(prev, curr)

            if timestamp <= slowThreshold:
                fast_mat[xx.flatten(), yy.flatten()] += 1
            else:
                slow_mat[xx.flatten(), yy.flatten()] += 1

            prevHit = newKeys
            timestamp = 0
    
    fast_mat = swapLdk(fast_mat)
    slow_mat = swapLdk(slow_mat)
    
    raw = [fast_mat, slow_mat]
    
    fast_mat = normKD(fast_mat)
    slow_mat = normKD(slow_mat)
    
    fast_norm = fast_mat / max(1, np.sum(fast_mat))
    slow_norm = slow_mat / max(1, np.sum(slow_mat))
    feat = np.stack([fast_norm, slow_norm], axis = 0).astype(np.float64)
#     feat = fast_mat
#     feat = feat / np.sum(feat)
    
    return feat, raw

class GaussianKde(scipy.stats.gaussian_kde):
    """
    Drop-in replacement for gaussian_kde that adds the class attribute EPSILON
    to the covmat eigenvalues, to prevent exceptions due to numerical error.
    """

    EPSILON = 1e-10  # adjust this at will

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=1,
                                                         bias=False,
                                                         aweights=self.weights))
            # we're going the easy way here
            self._data_covariance += self.EPSILON * np.eye(
                len(self._data_covariance))
            self._data_inv_cov = np.linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2
        L = np.linalg.cholesky(self.covariance * 2 * np.pi)
        self._norm_factor = 2*np.log(np.diag(L)).sum()  # needed for scipy 1.5.2
        self.log_det = 2*np.log(np.diag(L)).sum()  # changed var name on 1.6.2

def getBase64(arr):
    test = arr
    test = (test / (np.max(test) - np.min(test)))
    test = cmap(test)
    im = Image.fromarray(np.uint8(test * 255))

    buffered = BytesIO()
    im.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return str(img_str)[2:-1]

def to_feats(file):
    if ".osr" in file:
        rep = Replay.from_path(os.path.join("replays/", file))
        feat, [raw_fast, raw_slow] = get_features(rep, slowThreshold)
        return (feat, {'Player': rep.username, 'fast': raw_fast, 'slow': raw_slow, 'file': file})
    return None

# cd D:\RP_ML\osutSR2\ ; conda activate osu
if __name__ == "__main__":

    multiprocessing.freeze_support()

    features = []
    cnt = 0

    mpN = 32
    pool = multiprocessing.Pool(mpN)

    all_files = list(os.listdir("replays/"))
    # all_files = all_files[:10]
    features = list(tqdm.tqdm(pool.imap_unordered(to_feats, all_files), total = len(all_files)))

    pickle.dump(features, open("out-features.pkl", "wb"))
    print("Features saved to out-features.pkl")

    print("Starting UMAP")

    reducer = umap.UMAP()
    Xs = [f[0].flatten() for f in features]
    Ys = [f[1] for f in features]
    emb = reducer.fit_transform(Xs)

    print("UMAP done, outputting data")

    Xs_KDE = np.transpose(np.array(Xs))
    print(Xs_KDE.shape)
    kernel = GaussianKde(np.unique(Xs_KDE, axis = 1))
    KDEresult = kernel(Xs_KDE)

    hex_colors_dic = {}
    rgb_colors_dic = {}
    hex_colors_only = []
    for name, hex in matplotlib.colors.cnames.items():
        hex_colors_only.append(hex)
        hex_colors_dic[name] = hex
        rgb_colors_dic[name] = matplotlib.colors.to_rgb(hex)

    cmap = plt.get_cmap("viridis")

    data = {
        'UMAP-x': emb[:, 0], 
        'UMAP-y': emb[:, 1], 
        'Player': [Y['Player'] for Y in Ys],
        'Player-short': [Y['Player'][:5] for Y in Ys],
        'Color': [hex_colors_only[hash(Y['Player']) % len(hex_colors_only)] for Y in Ys],
        'file': [Y['file'] for Y in Ys],
        'KDE': [KDEresult[i] for i in range(len(Ys))],
    #     'fast': [f"\n{Y['fast']}" for Y in Ys],
    #     'slow': [f"\n{Y['slow']}" for Y in Ys],
        'fast': [getBase64(Y['fast']) for Y in Ys],
        'slow': [getBase64(Y['slow']) for Y in Ys],
    }
    df = pd.DataFrame(data)

    df.to_csv("test_new_dkTnorm.csv")
    print("Done, saved to test_new_dkTnorm.csv")
