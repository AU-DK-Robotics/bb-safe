import numpy as np
from PIL import Image
from snow import apply_snow
import matplotlib.pyplot as plt

# Random generator
rand_gen = np.random.default_rng()

# Initialize test pixel values
test_arr = np.zeros((1080,1920,3),np.uint8)

# Make half the image brighter
test_arr[:,961:1920,:] = 127

# Apply snow effect to test pixel values
test_snow_arr, snow_arr = apply_snow(rand_gen,test_arr)

# Convert pixel values to image
test_snow_img = Image.fromarray(test_snow_arr)
snow_img = Image.fromarray(snow_arr)

# Show the output image with default application
# test_snow_img.show()
# snow_img.show()

test_snow_dark_arr = test_snow_arr[:,0:960,:]
test_snow_light_arr = test_snow_arr[:,961:1920,:]

# Analyze statistical properties of the output image
mean_dark  = np.mean(test_snow_dark_arr)
mean_light = np.mean(test_snow_light_arr)
std_dark  = np.std(test_snow_dark_arr)
std_light = np.std(test_snow_light_arr)
var_dark  = np.var(test_snow_dark_arr)
var_light = np.var(test_snow_light_arr)

print(f"Dark pixels: {mean_dark} +/- {std_dark}")
print(f"Light pixels: {mean_light} +/- {std_light} ")

# Plot histograms
dark_count, dark_bins = np.histogram(test_snow_dark_arr,bins=256)
light_count, light_bins = np.histogram(test_snow_light_arr,bins=256)

plt.figure()
plt.stairs(dark_count,dark_bins)
plt.stairs(light_count,light_bins)
plt.show()
