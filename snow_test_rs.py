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
pi_gain = 5
pi_pixel_area = 1.12**2
rs_pixel_area = (1.4*(1080/720))**2
pi_rs_incomplete_conversion_factor = rs_pixel_area/(pi_pixel_area*pi_gain*pi_t_exp)

print(f"Conversion factor neglecting RS gain, RS exposure time: {pi_rs_incomplete_conversion_factor}")

gamma_rate = 600/60 # Gy/min
camera = RealSenseInterface(snow_factor=pi_rs_incomplete_conversion_factor,snow_rate=gamma_rate)

snow_img, _, _, color_frame_orig  = camera.get_frames()

cv2.imwrite(out_path / "snow.png", snow_img)
cv2.imwrite(out_path / "orig.png", color_frame_orig)

print("Done")


# Analyze statistical properties of the output image
mean  = np.mean(snow_img)
std  = np.std(snow_img,mean=mean,ddof=1)
print(f"Pixels: mean {mean}, std {std}")
