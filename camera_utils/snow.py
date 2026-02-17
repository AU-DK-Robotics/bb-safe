import numpy as np

def apply(img,mean):

    # Make a random generator
    rand_gen = np.random.default_rng()

    # Use floats for calculation
    img = img.astype(np.float64)

    # Draw samples from the Poisson distribution with
    # lam expected number of events per image pixel
    # snow = rand_gen.normal(loc=mean,scale=std,size=img.shape)
    snow = rand_gen.poisson(lam=mean,size=img.shape)

    # Normalize snow and pixel values
    snow = snow/255
    img = img/255

    # Reduce snow intensity in bright areas
    snow = snow*(1-img)

    # Add the random samples to the input image data
    snow_img = img + snow

    # Clip to max allowed pixel value (uint8: 255)
    snow_img = np.clip(snow_img,max=1.0,out=snow_img)
    snow = np.clip(snow,max=1.0,out=snow)

    # Convert back to integers
    snow_img = (255*snow_img).astype(np.uint8)
    snow = (255*snow).astype(np.uint8)

    return snow_img, snow

def model(dose_rate):
    # factor: sensitivity conversion from Picam R2 to RealSense D435
    # snow_rate: gamma radiation variable for linear regression (Gy/min)

    # factor = factor*gain
    # print(f"Final factor: {factor} (Gain: {gain})")

    # print(f"Exposure time: {t_exp} sec")

    # print(f"Dose rate: {dose_rate} Gy/min")

    mean_slope = 66.034 * (dose_rate ** 0.28)

    # mean = factor*(mean_slope*t_exp)

    # print(f"Resulting Poisson distribution expected value: {mean}")

    # std_0 = 0.81
    # std_slope = 6.7
    # std = (t_exp/1.3)*factor*(std_0 + std_slope*np.sqrt(self.snow_rate))

    return mean_slope
