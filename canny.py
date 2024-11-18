# Megan Danh
# CAP 5415: Computer Vision
# Fall 2024
# Programming Assignment 1: Canny Edge Detector

# Import required libraries 
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from matplotlib.image import imread

# Generating a 1D Gaussian mask
def gmask(sigma, size):
    xrange = np.linspace(-size//2, size //2, size, dtype=np.float32)
    G = np.exp(-(xrange ** 2)/(2*sigma ** 2)) # G(x) = e^(-(x^2)/(2*sigma^2))
    G = G/np.sum(G)
    return G

# Generating the first derivative of the Gaussian in the x and y directions (derivative of equation in gmask)
def firstGDerivative(sigma, size):
    xrange = np.linspace(-size//2, size //2, size, dtype=np.float32)
    Gx = (-xrange/(sigma**2))*np.exp(-(xrange ** 2)/(2*sigma ** 2)) # G(x) = (-x/sigma^2) * e^(-(x^2)/(2*sigma^2))
    Gx = Gx.reshape(1, -1) # row vector
    Gx = Gx/np.sum(np.abs(Gx))  # Normalize Gx
    
    Gy = (-xrange/(sigma**2))*np.exp(-(xrange ** 2)/(2*sigma ** 2))
    Gy = Gy.reshape(-1, 1) # column vector
    Gy = Gy / np.sum(np.abs(Gy))  # Normalize Gy

    return Gx, Gy

# Convolving function
def convolution(img, kernel):
    ih, iw = img.shape
    kh, kw = kernel.shape

    padh = kh //2
    padw = kw // 2

    # Creating a padded image so kernel can cover each pixel (no edges left out)
    pad_img = np.pad(img, ((padh, padh), (padw, padw)), mode='constant', constant_values=0)
    
    output_img = np.zeros_like(img)

    for i in range(ih):
        for j in range(iw):
            neighborhood = pad_img[i:i + kh, j:j + kw] # Defining the neighborhood
            output_img[i, j] = np.sum(neighborhood * kernel) # Summing to convolve

    return output_img

# Find gradient, or Ix_prime and Iy_prime
def find_gradient(i, Gx, Gy):
    Ix_prime = convolution(i, Gx)
    Iy_prime = convolution(i, Gy)

    return Ix_prime, Iy_prime

# Find magnitude
def find_magnitude(Ix_prime, Iy_prime):
    magnitude = np.hypot(Ix_prime, Iy_prime)
    magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude)) # Normalize
    return magnitude

# Find gradient direction
def find_gradient_direction(Ix, Iy): 
    gradient_direction = np.arctan2(Iy, Ix)
    gradient_direction = np.rad2deg(gradient_direction) # Convert to degrees
    return gradient_direction

# Find non-maximum suppression
def non_maximum_suppression(magnitude, gradient_direction):
    ih, iw = magnitude.shape 
    nms_img = np.zeros((ih, iw), dtype=np.float32)
        
    for i in range(1, ih-1):
        for j in range(1, iw-1):
            q, r = -np.inf, -np.inf  

            # Angle 0° (horizontal edges)
            if (-22.5 <= gradient_direction[i, j] < 22.5) or (-180.0 <= gradient_direction[i, j] < -157.5) or (157.5 <= gradient_direction[i, j] <= 180.0):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # Angle 45° (Top-right and bottom-left)
            elif (22.5 <= gradient_direction[i, j] < 67.5) or (-157.5 <= gradient_direction[i, j] < -112.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # Angle 90° (vertical edges)
            elif (67.5 <= gradient_direction[i, j] < 112.5) or (-112.5 <= gradient_direction[i, j] < -67.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # Angle 135° (Top-left and bottom-right)
            elif (112.5 <= gradient_direction[i, j] < 157.5) or (-67.5 <= gradient_direction[i, j] < -22.5):  
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]
            
            # If pixel's value is high enough to be an edge, keep
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                nms_img[i, j] = magnitude[i, j]
            
            # Otherwise, suppress
            else:
                nms_img[i, j] = 0  

    nms_img = nms_img / np.max(np.abs(nms_img)) # Normalize the magnitudes

    return nms_img

# Implement hysteresis thresholding
def hysteresis_thresholding(magnitude, low_thres, high_thres):
    strong = (magnitude > high_thres) # Edges larger than high threshold are strong
    weak = (magnitude >= low_thres) & (magnitude <= high_thres) # Edges smaller than low and high threshold are weak
    edges = np.zeros_like(magnitude, dtype=bool)
    
    # Connects weak edges to strong ones
    def trace(i, j):
        if weak[i, j]:
            edges[i, j] = True # If weak edge is adjacent to strong edge so it counts
            weak[i, j] = False # Not a weak edge is adjacent to a strong edge
            # Loops through all possible pixel neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if 0 <= i+di < magnitude.shape[0] and 0 <= j+dj < magnitude.shape[1]:
                        trace(i+di, j+dj) # Recursion to check weak edges

    # Loops through image and examines each pixel
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            if strong[i, j]:
                edges[i, j] = True
                trace(i, j)
    
    return edges

# Canny Edge Detector
def canny_edge(img_src, sigma, gaus_width, low_thres, high_thres):
    image = np.array(PIL.Image.open(img_src).convert('L')).astype(np.float32)
    i = np.array(image)

    # Creating the 1D Gaussian masks 
    G = gmask(sigma, gaus_width)
    Gx, Gy = firstGDerivative(sigma, gaus_width)
    
    # Using Gaussian filter to smooth
    smoothed_i = convolution(i, G.reshape(1, -1))
    Ix = smoothed_i # Shown in intermediate results
    smoothed_i = convolution(smoothed_i, G.reshape(-1, 1))
    Iy = smoothed_i # Shown in intermediate results

    # Find gradients from the smoothed image
    Ix_prime, Iy_prime = find_gradient(smoothed_i, Gx, Gy)
    
    # Computations with Ix' and Iy' for magnitude and gradient direction
    magnitude = find_magnitude(Ix_prime, Iy_prime)
    grad_dir = find_gradient_direction(Ix_prime, Iy_prime)
    nms_img = non_maximum_suppression(magnitude, grad_dir)
    hys_img = hysteresis_thresholding(nms_img, low_thres, high_thres)
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    # Display for Ix
    axs[0, 0].imshow(Ix, cmap='gray')
    axs[0, 0].set_title('Ix')

    # Display for Iy
    axs[0, 1].imshow(Iy, cmap='gray')
    axs[0, 1].set_title('Iy')

    # Display for Ix_prime
    axs[0, 2].imshow(Ix_prime, cmap='gray') 
    axs[0, 2].set_title('Ix_prime')

    # Display for Iy_prime
    axs[1, 0].imshow(Iy_prime, cmap='gray')
    axs[1, 0].set_title('Iy_prime')

    # Display for magnitude image
    axs[1, 1].imshow(magnitude, cmap='gray')
    axs[1, 1].set_title('Magnitude')

    # Display for non-maximum suppression image
    axs[1, 2].imshow(nms_img, cmap='gray')
    axs[1, 2].set_title('Non-Maximum Suppression Image')

    # Display for hysteresis thresholding
    axs[2, 1].imshow(hys_img, cmap='gray')
    axs[2, 1].set_title('Hysteresis Thresholding (Final)')


    # Formatting display
    for ax in axs.flat:
        ax.axis('off') 

    plt.tight_layout()
    fig.canvas.manager.set_window_title('Canny Edge Detector (Intermediate and Final Results)')
    plt.show()

# Image is placed in the same directory as edge detector 
# Ideal parameters:
#   Sigma = 1.8
#   Kernel size = 7
#   Low threshold = 0.06
#   High threshold = 0.19
canny_edge('einstein.jpg', 1.8, 7, 0.06, 0.19)

# Canny edge detection for other images in the folder
#
# canny_edge('bird.jpg', 1.8, 7, 0.06, 0.19) 
# canny_edge('parthenon.jpg', 1.8, 7, 0.06, 0.19)