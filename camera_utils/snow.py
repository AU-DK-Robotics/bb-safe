import numpy as np

def apply(img,lam=127):

    # Make a random generator
    rand_gen = np.random.default_rng()

    # Use floats for calculation
    img = img.astype(np.float16)

    # Draw samples from the Poisson distribution with
    # lam expected number of events per image pixel
    snow = rand_gen.poisson(lam=lam,size=img.shape)

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
