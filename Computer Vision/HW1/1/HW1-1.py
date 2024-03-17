import numpy as np
import cv2
import scipy.signal
from numpy import exp, pi, square, sqrt, arctan, divide, uint8

# A. Corner Detection

## a. Gaussian Smooth
def gaussian_smooth(image, sigma, k_size):
    cx, cy =  (0 + k_size-1) / 2, (0 + k_size - 1) / 2
    x, y =  np.meshgrid(np.arange(k_size) - cx, np.arange(k_size) - cy)
    gaussian_kernel = exp(-(square(x) + square(y)) / (2 * square(sigma))) / (2 * pi * square(sigma)) # G(x,y)
    gaussian_kernel = gaussian_kernel/gaussian_kernel.sum()
    # return gaussian_kernel
    image_gauss = scipy.signal.convolve2d(image, gaussian_kernel, mode = 'same') # I x G
    return image_gauss

## b. Intensity Gradient (Sobel edge detection)
def sobel_edge_detection(image, threshold):
    # Ignore divide and invalid exception
    np.seterr(divide = 'ignore', invalid = 'ignore')
    # Sobel operator
    Hx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Hy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    gradient_x = scipy.signal.convolve2d(image, Hx, mode='same')
    gradient_y = scipy.signal.convolve2d(image, Hy, mode='same')
    
    gradient_magnitude = sqrt(square(gradient_x) + square(gradient_y)) # find gradient magnitude
    gradient_direction = arctan(divide(gradient_x, gradient_y)) # find gradient direction

    _, mask = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)
    image_map = np.zeros((gradient_direction.shape[0], gradient_direction.shape[1], 3), dtype=np.uint8)

    # BGR color
    red = np.array([0, 0, 255])
    orange = np.array([0, 127, 255]) 
    green = np.array([0, 255, 0])
    yellow = np.array([0, 255, 255])
    blue = np.array([255, 0, 0])
    purple = np.array([255, 0, 127])
    pink = np.array([255, 0, 255])
    cyan = np.array([255, 255, 0])

    # mark color based on gradient_direction
    image_map[(mask == 255) & (gradient_direction > pi/3) & (gradient_direction <= pi/2) ] = red
    image_map[(mask == 255) & (gradient_direction > pi/6) & (gradient_direction <= pi/3)] = orange
    image_map[(mask == 255) & (gradient_direction > 0) & (gradient_direction <= pi/6)] = green
    image_map[(mask == 255) & (gradient_direction > -pi/6) & (gradient_direction <= -0)] = yellow
    image_map[(mask == 255) & (gradient_direction > -pi/3) & (gradient_direction <= -pi/6)] = blue
    image_map[(mask == 255) & (gradient_direction > -pi/2) & (gradient_direction <= -pi/3)] = purple

    return gradient_magnitude, image_map, gradient_x, gradient_y

## c. Structure Tensor
def structure_tensor(gussian_x, gussian_y, image, window_size): 
    k = 0.04 # k value from unit 2 p.68
    Ixx = square(gussian_x)
    Ixy = gussian_x * gussian_y
    Iyy = square(gussian_y)

    height, width = image.shape[0], image.shape[1]
    offset = int(window_size) // 2
    struct_tensor = np.zeros([height,width])
	
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])/(window_size**2)
            Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])/(window_size**2)
            Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])/(window_size**2)
            # Calculate determinant, trace and cornerness
            det = (Sxx * Syy) - square(Sxy)
            trace = Sxx + Syy
            r = det - k * square(trace)

            struct_tensor[y,x] = r

    return struct_tensor

## d. Non-maximal Suppression
def nms(image, struct, corner_threshold):
    corner = np.copy(image)
    for y, row in enumerate(struct):
        for x, pixel in enumerate(row):
            if pixel.all() > corner_threshold:
                cv2.circle(corner, (x, y), 1, (0, 255, 0))
    return corner

# Read 1a_notredame image and 1b_notredame image
img1 = cv2.imread('chessboard-hw1.jpg')
img2 = cv2.imread('1a_notredame.jpg')

# Convert chessboard-hw1 image and 1a_notredame image to gray scale
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img1_guassian1 = gaussian_smooth(img1_gray, sigma = 5, k_size = 5)
img1_guassian2 = gaussian_smooth(img1_gray, sigma = 5, k_size = 10)
img2_guassian1 = gaussian_smooth(img2_gray, sigma = 5, k_size = 5)
img2_guassian2 = gaussian_smooth(img2_gray, sigma = 5, k_size = 10)

# Save chessboard-hw1 and 1a_notredame images after gaussian smooth into normal folder
cv2.imwrite('./output/normal/chessboard-hw1 after gaussian smooth (sigma = 5, kernel size = 5).jpg', img1_guassian1)
cv2.imwrite('./output/normal/chessboard-hw1 after gaussian smooth (sigma = 5, kernel size = 10).jpg', img1_guassian2)
cv2.imwrite('./output/normal/1a_notredame after gaussian smooth (sigma = 5, kernel size = 5).jpg', img2_guassian1)
cv2.imwrite('./output/normal/1a_notredame after gaussian smooth (sigma = 5, kernel size = 10).jpg', img2_guassian2)

magnitude_img1_guassian1, direction_img1_guassian1, img1_guassian1_gradient_x, img1_guassian1_gradient_y = sobel_edge_detection(img1_guassian1, 50)
magnitude_img1_guassian2, direction_img1_guassian2, img1_guassian2_gradient_x, img1_guassian2_gradient_y = sobel_edge_detection(img1_guassian2, 50)
magnitude_img2_guassian1, direction_img2_guassian1, img2_guassian1_gradient_x, img2_guassian1_gradient_y = sobel_edge_detection(img2_guassian1, 50)
magnitude_img2_guassian2, direction_img2_guassian2, img2_guassian2_gradient_x, img2_guassian2_gradient_y = sobel_edge_detection(img2_guassian2, 50)

# Save chessboard-hw1 and 1a_notredame magnitude and direction after gaussian smooth into normal folder
cv2.imwrite('./output/normal/chessboard-hw1 magnitude after gaussian smooth (sigma = 5, kernel size = 5).jpg', magnitude_img1_guassian1)
cv2.imwrite('./output/normal/chessboard-hw1 direction after gaussian smooth (sigma = 5, kernel size = 5).jpg', direction_img1_guassian1)
cv2.imwrite('./output/normal/chessboard-hw1 magnitude after gaussian smooth (sigma = 5, kernel size = 10).jpg', magnitude_img1_guassian2)
cv2.imwrite('./output/normal/chessboard-hw1 direction after gaussian smooth (sigma = 5, kernel size = 10).jpg', direction_img1_guassian2)
cv2.imwrite('./output/normal/1a_notredame magnitude after gaussian smooth (sigma = 5, kernel size = 5).jpg', magnitude_img2_guassian1)
cv2.imwrite('./output/normal/1a_notredame direction after gaussian smooth (sigma = 5, kernel size = 5).jpg', direction_img2_guassian1)
cv2.imwrite('./output/normal/1a_notredame magnitude after gaussian smooth (sigma = 5, kernel size = 10).jpg', magnitude_img2_guassian2)
cv2.imwrite('./output/normal/1a_notredame direction after gaussian smooth (sigma = 5, kernel size = 10).jpg', direction_img2_guassian2)

img1_guassian2_win3_st = structure_tensor(img1_guassian2_gradient_x, img1_guassian2_gradient_y, img1, 3)
img1_guassian2_win5_st = structure_tensor(img1_guassian2_gradient_x, img1_guassian2_gradient_y, img1, 5)
img2_guassian2_win3_st = structure_tensor(img2_guassian2_gradient_x, img2_guassian2_gradient_y, img2, 3)
img2_guassian2_win5_st = structure_tensor(img2_guassian2_gradient_x, img2_guassian2_gradient_y, img2, 5)

# Save chessboard-hw1 and 1a_notredame magnitude and direction after gaussian smooth into normal folder
cv2.imwrite('./output/normal/chessboard-hw1 structure tensor (sigma = 5, kernel size = 10, window size 3x3).jpg', img1_guassian2_win3_st)
cv2.imwrite('./output/normal/chessboard-hw1 structure tensor (sigma = 5, kernel size = 10, window size 5x5).jpg', img1_guassian2_win5_st)
cv2.imwrite('./output/normal/1a_notredame structure tensor (sigma = 5, kernel size = 10, window size 3x3).jpg', img2_guassian2_win3_st)
cv2.imwrite('./output/normal/1a_notredame structure tensor (sigma = 5, kernel size = 10, window size 5x5).jpg', img2_guassian2_win5_st)

img1_guassian2_win3_st = cv2.imread('./output/normal/chessboard-hw1 structure tensor (sigma = 5, kernel size = 10, window size 3x3).jpg')
img1_guassian2_win5_st = cv2.imread('./output/normal/chessboard-hw1 structure tensor (sigma = 5, kernel size = 10, window size 5x5).jpg')
img2_guassian2_win3_st = cv2.imread('./output/normal/1a_notredame structure tensor (sigma = 5, kernel size = 10, window size 3x3).jpg')
img2_guassian2_win5_st = cv2.imread('./output/normal/1a_notredame structure tensor (sigma = 5, kernel size = 10, window size 5x5).jpg')

nms_img1_guassian2_win3 = nms(img1, img1_guassian2_win3_st, 0.1)
nms_img1_guassian2_win5 = nms(img1, img1_guassian2_win5_st, 0.1)
nms_img2_guassian2_win3 = nms(img2, img2_guassian2_win3_st, 0.1)
nms_img2_guassian2_win5 = nms(img2, img2_guassian2_win5_st, 0.1)

cv2.imwrite('./output/normal/chessboard-hw1 after nms (sigma = 5, kernel size = 10, window size 3x3).jpg', nms_img1_guassian2_win3)
cv2.imwrite('./output/normal/chessboard-hw1 after nms (sigma = 5, kernel size = 10, window size 5x5).jpg', nms_img1_guassian2_win5)
cv2.imwrite('./output/normal/1a_notredame after nms (sigma = 5, kernel size = 10, window size 3x3).jpg', nms_img2_guassian2_win3)
cv2.imwrite('./output/normal/1a_notredame after nms (sigma = 5, kernel size = 10, window size 5x5).jpg', nms_img2_guassian2_win5)

# B. Experiments (Rotate and Scale)

# Rotate image by 30°
def rotate(image, a = 30):
    h, w = image.shape[0], image.shape[1] # height, width
    c = w/2, h/2 # center
    image_rotated = cv2.getRotationMatrix2D(c, angle = a, scale = 1)
    image_rotated = cv2.warpAffine(image, image_rotated, (w, h))
    return image_rotated

# Scale image by 0.5
def scale(image, s = 0.5):    
    h, w = image.shape[0], image.shape[1] # height, width
    c = w/2, h/2 # center
    image_rotated = cv2.getRotationMatrix2D(c, angle = 0, scale = s)
    image_rotated = cv2.warpAffine(image, image_rotated, (w, h))
    return image_rotated

img1_rotate_30 = rotate(img1)
img2_rotate_30 = rotate(img2)
img1_scale_half = scale(img1)
img2_scale_half = scale(img2)

# Write chessboard-hw1 and 1a_notredame images after rotate and scale into transformed folder.
cv2.imwrite('./output/transformed/chessboard-hw1 (rotated by 30 degree).jpg', img1_rotate_30)
cv2.imwrite('./output/transformed/1a_notredame (rotated by 30 degree).jpg', img2_rotate_30) 
cv2.imwrite('./output/transformed/chessboard-hw1 (scaled by 0.5).jpg', img1_scale_half)
cv2.imwrite('./output/transformed/1a_notredame (scaled by 0.5).jpg', img2_scale_half) 

img1_rotate_30_gray = cv2.cvtColor(img1_rotate_30, cv2.COLOR_BGR2GRAY)
img2_rotate_30_gray = cv2.cvtColor(img2_rotate_30, cv2.COLOR_BGR2GRAY)
img1_scale_half_gray = cv2.cvtColor(img1_scale_half, cv2.COLOR_BGR2GRAY)
img2_scale_half_gray = cv2.cvtColor(img2_scale_half, cv2.COLOR_BGR2GRAY)

## a. Gaussian Smooth
img1_rotate_30_guassian1 = gaussian_smooth(img1_rotate_30_gray, sigma = 5, k_size = 5)
img2_rotate_30_guassian1 = gaussian_smooth(img2_rotate_30_gray, sigma = 5, k_size = 5)
img1_rotate_30_guassian2 = gaussian_smooth(img1_rotate_30_gray, sigma = 5, k_size = 10)
img2_rotate_30_guassian2 = gaussian_smooth(img2_rotate_30_gray, sigma = 5, k_size = 10)
img1_scale_half_guassian1 = gaussian_smooth(img1_scale_half_gray, sigma = 5, k_size = 5)
img2_scale_half_guassian1 = gaussian_smooth(img2_scale_half_gray, sigma = 5, k_size = 5)
img1_scale_half_guassian2 = gaussian_smooth(img1_scale_half_gray, sigma = 5, k_size = 10)
img2_scale_half_guassian2 = gaussian_smooth(img2_scale_half_gray, sigma = 5, k_size = 10)

cv2.imwrite('./output/transformed/chessboard-hw1 (rotated by 30 degree) after gaussian smooth (sigma = 5, kernel size = 5).jpg', img1_rotate_30_guassian1)
cv2.imwrite('./output/transformed/1a_notredame (rotated by 30 degree) after gaussian smooth (sigma = 5, kernel size = 5).jpg', img2_rotate_30_guassian1)
cv2.imwrite('./output/transformed/chessboard-hw1 (rotated by 30 degree) after gaussian smooth (sigma = 5, kernel size = 10).jpg', img1_rotate_30_guassian2)
cv2.imwrite('./output/transformed/1a_notredame (rotated by 30 degree) after gaussian smooth (sigma = 5, kernel size = 10).jpg', img2_rotate_30_guassian2)
cv2.imwrite('./output/transformed/chessboard-hw1 (scaled by 0.5) after gaussian smooth (sigma = 5, kernel size = 5).jpg', img1_scale_half_guassian1)
cv2.imwrite('./output/transformed/1a_notredame (scaled by 0.5) after gaussian smooth (sigma = 5, kernel size = 5).jpg', img2_scale_half_guassian1) 
cv2.imwrite('./output/transformed/chessboard-hw1 (scaled by 0.5) after gaussian smooth (sigma = 5, kernel size = 10).jpg', img1_scale_half_guassian2)
cv2.imwrite('./output/transformed/1a_notredame (scaled by 0.5) after gaussian smooth (sigma = 5, kernel size = 10).jpg', img2_scale_half_guassian2) 

## b. Intensity Gradient
magnitude_img1_rotate_30_guassian1, direction_img1_rotate_30_guassian1, img1_guassian1_rotate_30_gradient_x, img1_guassian1_rotate_30_gradient_y = sobel_edge_detection(img1_rotate_30_guassian1, 50)
magnitude_img2_rotate_30_guassian1, direction_img2_rotate_30_guassian1, img2_guassian1_rotate_30_gradient_x, img2_guassian1_rotate_30_gradient_y = sobel_edge_detection(img2_rotate_30_guassian1, 50)
magnitude_img1_scale_half_guassian1, direction_img1_scale_half_guassian1, img1_guassian1_scale_half_gradient_x, img1_guassian1_scale_half_gradient_y = sobel_edge_detection(img1_scale_half_guassian1, 50)
magnitude_img2_scale_half_guassian1, direction_img2_scale_half_guassian1, img2_guassian1_scale_half_gradient_x, img2_guassian1_scale_half_gradient_y = sobel_edge_detection(img2_scale_half_guassian1, 50)

# Write chessboard-hw1 and 1a_notredame magnitude and direction after gaussian smooth into transformed folder.
cv2.imwrite('./output/transformed/chessboard-hw1 (rotated by 30 degree) magnitude after gaussian smooth.jpg', magnitude_img1_rotate_30_guassian1)
cv2.imwrite('./output/transformed/chessboard-hw1 (rotated by 30 degree) direction after gaussian smooth.jpg', direction_img1_rotate_30_guassian1)
cv2.imwrite('./output/transformed/1a_notredame (rotated by 30 degree) magnitude after gaussian smooth.jpg', magnitude_img2_rotate_30_guassian1)
cv2.imwrite('./output/transformed/1a_notredame (rotated by 30 degree) direction after gaussian smooth.jpg', direction_img2_rotate_30_guassian1)
cv2.imwrite('./output/transformed/chessboard-hw1 (scaled by 0.5) magnitude after gaussian smooth.jpg', magnitude_img1_scale_half_guassian1)
cv2.imwrite('./output/transformed/chessboard-hw1 (scaled by 0.5) direction after gaussian smooth.jpg', direction_img1_scale_half_guassian1)
cv2.imwrite('./output/transformed/1a_notredame (scaled by 0.5) magnitude after gaussian smooth.jpg', magnitude_img2_scale_half_guassian1)
cv2.imwrite('./output/transformed/1a_notredame (scaled by 0.5) direction after gaussian smooth.jpg', direction_img2_scale_half_guassian1)

## c. Structure Tensor
img1_rotate_30_st_win3 = structure_tensor(img1_guassian1_rotate_30_gradient_x, img1_guassian1_rotate_30_gradient_y, img1_rotate_30, 3)
img2_rotate_30_st_win3 = structure_tensor(img2_guassian1_rotate_30_gradient_x, img2_guassian1_rotate_30_gradient_y, img2_rotate_30, 3)
img1_rotate_30_st_win5 = structure_tensor(img1_guassian1_rotate_30_gradient_x, img1_guassian1_rotate_30_gradient_y, img1_rotate_30, 5)
img2_rotate_30_st_win5 = structure_tensor(img2_guassian1_rotate_30_gradient_x, img2_guassian1_rotate_30_gradient_y, img2_rotate_30, 5)
img1_scale_half_st_win3 = structure_tensor(img1_guassian1_scale_half_gradient_x, img1_guassian1_scale_half_gradient_y, img1_scale_half, 3)
img2_scale_half_st_win3 = structure_tensor(img2_guassian1_scale_half_gradient_x, img2_guassian1_scale_half_gradient_y, img2_scale_half, 3)
img1_scale_half_st_win5 = structure_tensor(img1_guassian1_scale_half_gradient_x, img1_guassian1_scale_half_gradient_y, img1_scale_half, 5)
img2_scale_half_st_win5 = structure_tensor(img2_guassian1_scale_half_gradient_x, img2_guassian1_scale_half_gradient_y, img2_scale_half, 5)

# Write chessboard-hw1 and 1a_notredame magnitude and direction after gaussian smooth into transformed folder
cv2.imwrite('./output/transformed/chessboard-hw1 (rotated by 30 degree) structure tensor (window size 3x3).jpg', img1_rotate_30_st_win3)
cv2.imwrite('./output/transformed/1a_notredame (rotated by 30 degree) structure tensor (window size 3x3).jpg', img2_rotate_30_st_win3)
cv2.imwrite('./output/transformed/chessboard-hw1 (rotated by 30 degree) structure tensor (window size 5x5).jpg', img1_rotate_30_st_win5)
cv2.imwrite('./output/transformed/1a_notredame (rotated by 30 degree) structure tensor (window size 5x5).jpg', img2_rotate_30_st_win5)
cv2.imwrite('./output/transformed/chessboard-hw1 (scaled by 0.5) structure tensor (window size 3x3).jpg', img1_scale_half_st_win3)
cv2.imwrite('./output/transformed/1a_notredame (scaled by 0.5) structure tensor (window size 3x3).jpg', img2_scale_half_st_win3)
cv2.imwrite('./output/transformed/chessboard-hw1 (scaled by 0.5) structure tensor (window size 5x5).jpg', img1_scale_half_st_win5)
cv2.imwrite('./output/transformed/1a_notredame (scaled by 0.5) structure tensor (window size 5x5).jpg', img2_scale_half_st_win5)

img1_rotate_30_st = cv2.imread('./output/transformed/chessboard-hw1 (rotated by 30 degree) structure tensor (window size 5x5).jpg')
img2_rotate_30_st = cv2.imread('./output/transformed/1a_notredame (rotated by 30 degree) structure tensor (window size 5x5).jpg')
img1_scale_half_st = cv2.imread('./output/transformed/chessboard-hw1 (scaled by 0.5) structure tensor (window size 5x5).jpg')
img2_scale_half_st = cv2.imread('./output/transformed/1a_notredame (scaled by 0.5) structure tensor (window size 5x5).jpg')

nms_img1_rotate_window5 = nms(img1_rotate_30, img1_rotate_30_st, 0.1)
nms_img2_rotate_window5 = nms(img2_rotate_30, img2_rotate_30_st, 0.1)
nms_img1_scale_window5 = nms(img1_scale_half, img1_scale_half_st, 0.1)
nms_img2_scale_window5 = nms(img2_scale_half, img2_scale_half_st, 0.1)

cv2.imwrite('./output/transformed/chessboard-hw1 (rotated by 30 degree) after nms (window size 5x5).jpg', nms_img1_rotate_window5)
cv2.imwrite('./output/transformed/1a_notredame (rotated by 30 degree) after nms (window size 5x5).jpg', nms_img2_rotate_window5)
cv2.imwrite('./output/transformed/chessboard-hw1 (scaled by 0.5) after nms (window size 5x5).jpg', nms_img1_scale_window5)
cv2.imwrite('./output/transformed/1a_notredame (scaled by 0.5) after nms (window size 5x5).jpg', nms_img2_scale_window5)