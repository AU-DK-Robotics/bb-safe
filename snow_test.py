import numpy as np
from PIL import Image
from camera_utils import snow
import matplotlib.pyplot as plt

# Initialize test pixel values
test_arr = np.zeros((1080,1920,3),np.uint8)

# Make half the image brighter
test_arr[:,961:1920,:] = 127

# Apply snow effect to test pixel values
test_snow_arr, snow_arr = snow.apply(test_arr,lam=127)

# Convert pixel values to image
test_snow_img = Image.fromarray(test_snow_arr)
snow_img = Image.fromarray(snow_arr)

# Show the output image with default application
# test_snow_img.show()
# snow_img.show()

test_snow_dark_arr = test_snow_arr[:,0:960,:]
test_snow_light_arr = test_snow_arr[:,961:1920,:]

# Analyze statistical properties of the output image
# mean of a Poisson distribution should equal its variance (std^2)
mean_dark  = np.mean(test_snow_dark_arr)
mean_light = np.mean(test_snow_light_arr)
std_dark  = np.std(test_snow_dark_arr,mean=mean_dark,ddof=1)
std_light = np.std(test_snow_light_arr,mean=mean_light,ddof=1)

print(f"Dark pixels: mean {mean_dark}, variance {std_dark**2}")
print(f"Light pixels: mean {mean_light}, variance {std_light**2}")

# Plot histograms
dark_count, dark_bins = np.histogram(test_snow_dark_arr,bins=256)
light_count, light_bins = np.histogram(test_snow_light_arr,bins=256)

# plt.figure()
# plt.stairs(dark_count,dark_bins)
# plt.stairs(light_count,light_bins)
# plt.show()
