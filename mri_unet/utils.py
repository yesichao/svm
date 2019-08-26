import os
from glob import glob
from scipy import signal
import numpy as np
from scipy.signal import resample
import pywt
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
def get_image_files(sig_dir):
  fs = glob("{}/*.nii.gz".format(sig_dir))
  fs = [os.path.basename(filename) for filename in fs]
  return sorted(fs)