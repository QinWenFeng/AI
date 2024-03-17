import cv2
import numpy as np

# (a) Implement a function that estimates the homography matrix H that maps a set of interest points to a new set of interest points.
def homography(u, v):
    A = np.array([[u[0][0], u[0][1], 1, 0, 0, 0, -1 * u[0][0] * v[0][0], -1 * u[0][1] * v[0][0]],
                  [0, 0, 0, u[0][0], u[0][1], 1, -1 * u[0][0] * v[0][1], -1 * u[0][1] * v[0][1]],
                  [u[1][0], u[1][1], 1, 0, 0, 0, -1 * u[1][0] * v[1][0], -1 * u[1][1] * v[1][0]],
                  [0, 0, 0, u[1][0], u[1][1], 1, -1 * u[1][0] * v[1][1], -1 * u[1][1] * v[1][1]],
                  [u[2][0], u[2][1], 1, 0, 0, 0, -1 * u[2][0] * v[2][0], -1 * u[2][1] * v[2][0]],
                  [0, 0, 0, u[2][0], u[2][1], 1, -1 * u[2][0] * v[2][1], -1 * u[2][1] * v[2][1]],
                  [u[3][0], u[3][1], 1, 0, 0, 0, -1 * u[3][0] * v[3][0], -1 * u[3][1] * v[3][0]],
                  [0, 0, 0, u[3][0], u[3][1], 1, -1 * u[3][0] * v[3][1], -1 * u[3][1] * v[3][1]]
                ])

    b = np.array([[v[0][0]],
                  [v[0][1]],
                  [v[1][0]],
                  [v[1][1]],
                  [v[2][0]],
                  [v[2][1]],
                  [v[3][0]],
                  [v[3][1]]
                ])
 
    h = np.dot(np.linalg.inv(A), b) # solve Ah = b by multiply A^-1 to both side
    H = np.array([[h[0][0], h[1][0], h[2][0]],
                  [h[3][0], h[4][0], h[5][0]],
                  [h[6][0], h[7][0], 1]
                 ])
    # Return homography matrix.
    return H

# (b) Use bilnear interpolation.
def bilnear_interpolation(img, new_x, new_y):
    fx = round(new_x - int(new_x), 2)
    fy = round(new_y - int(new_y), 2)

    p = np.zeros((3,))
    try:
        p += (1 - fx) * (1 - fy) * img[int(new_y), int(new_x)]
        p += (1 - fx) * fy * img[int(new_y) + 1, int(new_x)]
        p += fx * (1 - fy) * img[int(new_y), int(new_x) + 1]
        p += fx * fy * img[int(new_y) + 1, int(new_x) + 1]
    except:
        pass

    return p

# (b) Use backward warping.
def backward_warping(image, output, homography_matrix):
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            new_pos = np.dot(homography_matrix, np.array([[x, y, 1]]).T)
            new_x, new_y = new_pos[0][0] / new_pos[2][0], new_pos[1][0] / new_pos[2][0]
            if new_x < image.shape[0] and new_x >= 0 and new_y < image.shape[0] and new_y >= 0:
                sample = bilnear_interpolation(image, new_x, new_y)
                output[y][x] = sample
    # Return image after backward warping.
    return output

# Draw selected line with green color and thickness of line = 4
def selected_line(image):
    image_with_line = np.copy(image)
    image_with_line = cv2.line(image_with_line, (435, 341), (890, 13), (0, 255, 0), 4) # top line
    image_with_line = cv2.line(image_with_line, (435, 341), (423, 817), (0, 255, 0), 4) # left line
    image_with_line = cv2.line(image_with_line, (890, 13), (894, 998), (0, 255, 0), 4) # right line
    image_with_line = cv2.line(image_with_line, (423, 817), (894, 998), (0, 255, 0), 4) # bottom line
    # Return image with four selected line.
    return image_with_line

src_img = cv2.imread('Delta-Building.jpg')
dst_img = np.zeros((1068, 1600, 3))

selected_img = selected_line(src_img)
# Write selected_img into output folder.
cv2.imwrite('./output/selected_img.jpg', selected_img)

h, w = src_img.shape[0], src_img.shape[1]

src_pts = np.array([[435, 341], [890, 13], 
                    [423, 817], [894, 998]])
dst_pts = np.array([[0, 0], [w-1, 0], 
                    [0, h-1], [w-1, h-1]])
H = homography(dst_pts, src_pts)
# Print homography matrix.
print('Homography matrix:')
print(H)

dst_img = backward_warping(src_img, dst_img, H)
# Write rectified_img into output folder.
cv2.imwrite('./output/rectified_img.jpg', dst_img)