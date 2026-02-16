import numpy as np
from PIL import Image
from camera_utils import snow
import matplotlib.pyplot as plt
from camera_utils.camera_interface_async import RealSenseInterfaceAsync as RealSenseInterface
import cv2
from pathlib import Path
import time

out_path = Path("snow_test_rs")
out_path.mkdir(parents=True,exist_ok=True)

pi_t_exp = 1.3
pi_gain = 1
pi_pixel_area = 1.12**2
rs_pixel_area = (1.4*(1080/720))**2
pi_rs_incomplete_conversion_factor = rs_pixel_area/(pi_pixel_area*pi_gain*pi_t_exp)

print(f"Conversion factor neglecting RS gain, RS exposure time: {pi_rs_incomplete_conversion_factor}")

gamma_rate = 600/60 # Gy/min
camera = RealSenseInterface(snow_factor=pi_rs_incomplete_conversion_factor,snow_rate=gamma_rate,recording_path=out_path)

t = time.perf_counter()
while time.perf_counter()-t < 5:
    pass
    # cv2.imshow("Camera views with simulated gamma-ray noise", color_frame)
    # cv2.waitKey(0)

print("Done")

# # Show the output image with default application
# test_snow_img.show()
# snow_img.show()

# test_snow_dark_arr = test_snow_arr[:,0:960,:]
# test_snow_light_arr = test_snow_arr[:,961:1920,:]

# # Analyze statistical properties of the output image
# # mean of a Poisson distribution should equal its variance (std^2)
# mean_dark  = np.mean(test_snow_dark_arr)
# mean_light = np.mean(test_snow_light_arr)
# std_dark  = np.std(test_snow_dark_arr,mean=mean_dark,ddof=1)
# std_light = np.std(test_snow_light_arr,mean=mean_light,ddof=1)

# print(f"Dark pixels: mean {mean_dark}, variance {std_dark**2}")
# print(f"Light pixels: mean {mean_light}, variance {std_light**2}")

# # Plot histograms
# dark_count, dark_bins = np.histogram(test_snow_dark_arr,bins=256)
# light_count, light_bins = np.histogram(test_snow_light_arr,bins=256)

# plt.figure()
# plt.stairs(dark_count,dark_bins)
# plt.stairs(light_count,light_bins)
# plt.show()
