import numpy as np
import matplotlib.pyplot as plt

img_hdr = np.load("/home/ubuntu/data_sota_disk/dataset/inverse_tone/AIM_ITM_TRAIN/merged_data/ldr_to_hdr_full_range_batch_full/im_000001_000001.npy")

img_hdr = img_hdr/1000
img_hdr = img_hdr.squeeze()
plt.imshow(img_hdr)
plt.show()
