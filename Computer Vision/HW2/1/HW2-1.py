import cv2
import numpy as np
import scipy.linalg

# (a) Implement the linear least-squares eight-point algorithm and return fundamental matrix.
def linear_least_squares_eight_point(x1, x2):
    # Build matrix A
    A_list = []
    for i in range(x1.shape[0]):
        u1 = x1[i][0]
        v1 = x1[i][1]
        u2 = x2[i][0]
        v2 = x2[i][1]
        A_list.append([u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1, 1])
    # Transform list to array
    A = np.array(A_list)
    # Compute linear least square solution                 
    U, S, V = scipy.linalg.svd(A, full_matrices = False, overwrite_a = True)
    F = V[-1].reshape(3, 3)

    # Constraint F
    # Make rank 2 by zeroing out last singular value                
    U, S, V = scipy.linalg.svd(F, full_matrices = False, overwrite_a = True)
    F = U @ np.diag([*S[:2], 0]) @ V
    # Return fundamental matrix
    return F

# (b) Implement the normalized eight-point algorithm and return fundamental matrix.
def normalized_eight_point(x1, x2):
    # Normalize coordinates in image1
    mean1 = np.mean(x1, axis = 0) # compute mean in 3 columns [x1_mean, y1_mean, 1]
    center1 = x1 - mean1 * np.ones(x1.shape)      
    square1 = center1 * center1
    distance1 = np.sqrt(square1[:, 0] + square1[:, 1])
    std1 = np.mean(distance1)
    T1_tmp = np.array([[1, 0, -mean1[0]], [0, 1, -mean1[1]], [0, 0, 1]])
    T2_tmp = np.array([[np.sqrt(2) / std1, 0, 0], [0, np.sqrt(2) / std1, 0], [0, 0, 1]])
    T1 = T2_tmp @ T1_tmp
    x1 = (T1 @ x1.T).T
    # Normalize coordinates in image2
    mean2 = np.mean(x2, axis = 0) # compute mean in 3 columns [x2_mean, y2_mean, 1]
    center2 = x2 - mean2 * np.ones(x2.shape)      
    square2 = center2 * center2
    distance2 = np.sqrt(square2[:, 0] + square2[:, 1])
    std2 = np.mean(distance2)
    T3_tmp = np.array([[1, 0, -mean2[0]], [0, 1, -mean2[1]], [0, 0, 1]])
    T4_tmp = np.array([[np.sqrt(2) / std2, 0, 0], [0, np.sqrt(2) / std2, 0], [0, 0, 1]])
    T2 = T4_tmp @ T3_tmp
    x2 = (T2 @ x2.T).T
    # Build matrix A
    A_list = []
    for i in range(x1.shape[0]):
        u1 = x1[i][0]
        v1 = x1[i][1]
        u2 = x2[i][0]
        v2 = x2[i][1]
        A_list.append([u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1, 1])
    # Transform list to array
    A = np.array(A_list)
    # Compute linear least square solution                 
    U, S, V = scipy.linalg.svd(A, full_matrices=False, overwrite_a=True)
    F = V[-1].reshape(3, 3)
    # Constraint F
    # Make rank 2 by zeroing out last singular value                
    U, S, V = scipy.linalg.svd(F, full_matrices=False, overwrite_a=True)
    F = U @ np.diag([*S[:2], 0]) @ V
    # Denormalization
    F = T2.T @ F @ T1
    # Return fundamental matrix
    return F 

# (c) Plot the epipolar lines for the given point correspondences determined by the fundamental matrices computed from (a) and (b). 
def plot_epipolar_line(F, image1, image2, point1, point2):  
    image1_copy = np.copy(image1)
    image2_copy = np.copy(image2)
    lines1 = np.zeros((len(point1), 3))
    lines2 = np.zeros((len(point2), 3))
    for i in range(len(point1)):
        # Find epipolar line.
        line1 = np.dot(F.T, point2[i].T)
        line2 = np.dot(F, point1[i].T)
        # Find point to draw epipolar line.
        x0_1, y0_1 = map(int, [0, -line1[2]/line1[1]])
        x1_1, y1_1 = map(int, [image1_copy.shape[1] , -(line1[2]+line1[0]*image1_copy.shape[1])/line1[1]])
        x0_2, y0_2 = map(int, [0, -line2[2]/line2[1]])
        x1_2, y1_2 = map(int, [image2_copy.shape[1] , -(line2[2]+line2[0]*image2_copy.shape[1])/line2[1]])
        # Draw epipolar line on image.
        image1_copy = cv2.line(image1_copy, (x0_1, y0_1), (x1_1, y1_1), (0,0,255), 3)
        image2_copy = cv2.line(image2_copy, (x0_2, y0_2), (x1_2, y1_2), (0,0,255), 3)
        # Find slope (m1) and y-intercept (c) of epipolar line.
        m1 = (y1_1 - y0_1) / (x1_1 - x0_1)
        c1 = y1_1 - m1 * x1_1
        lines1[i] = [m1, -1, c1]
        m2 = (y1_2 - y0_2) / (x1_2 - x0_2)
        c2 = y1_2 - m2 * x1_2
        lines2[i] = [m2, -1, c2]
    # Return image with epipolar line and epipolar.
    return image1_copy, image2_copy, lines1, lines2

# Marking given point on image.
def plot_point(image1, image2, point1, point2):
    image1_copy, image2_copy = np.copy(image1), np.copy(image2)
    for i in range(len(point1)):
        x1, y1 = point1[i][0], point1[i][1]
        x2, y2 = point2[i][0], point2[i][1]
        image1_copy = cv2.circle(image1_copy, (int(x1), int(y1)), 5, (255, 0, 0), -1)
        image2_copy = cv2.circle(image2_copy, (int(x2), int(y2)), 5, (255, 0, 0), -1)
    # Return image with given point
    return image1_copy, image2_copy

# Find distance between epipolar line and feature points. 
def line_to_point_distance(point, line):
    (a, b, c) = line
    (x, y, _) = point
    # Return distance between given point and epipolar line.
    return np.abs(a*x + b*y + c) / np.sqrt(a*a + b*b)

# (c) Determine the accuracy of the fundamental matrices by computing the average distance between the feature points and their corresponding epipolar lines.
def average_epipolar_distance(point1, line1, point2, line2):
    distance1 = 0
    distance2 = 0
    for i in range(point1.shape[0]):
        distance1 = distance1 + line_to_point_distance(point1[i], line1[i])
        distance2 = distance2 + line_to_point_distance(point2[i], line2[i])
    # Return average distance between given point and epipolar line.
    return distance1 / point1.shape[0], distance2 / point2.shape[0]

# Read image1 and image2.
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')
point1 = np.zeros((46, 3))
point2 = np.zeros((46, 3))

# Read given pt_2D_1.txt.
with open('pt_2D_1.txt', 'r') as file1:
    file1.readline()
    for i, line in enumerate(file1.readlines()):
        pt = line.split()
        point1[i, 0] = float(pt[0])
        point1[i, 1] = float(pt[1])
        point1[i, 2] = 1

# Read given pt_2D_2.txt.
with open('pt_2D_2.txt', 'r') as file2:
    file2.readline()
    for i, line in enumerate(file2.readlines()):
        pt = line.split()
        point2[i, 0] = float(pt[0])
        point2[i, 1] = float(pt[1])
        point2[i, 2] = 1

fundamental_matrix_a = linear_least_squares_eight_point(point1, point2)
fundamental_matrix_b = normalized_eight_point(point1, point2)

a_img1, a_img2, lines_a_img1, lines_a_img2 = plot_epipolar_line(fundamental_matrix_a, img1, img2, point1, point2)
a_img1, a_img2 = plot_point(a_img1, a_img2, point1, point2)
b_img1, b_img2, lines_b_img1, lines_b_img2 = plot_epipolar_line(fundamental_matrix_b, img1, img2, point1, point2)
b_img1, b_img2 = plot_point(b_img1, b_img2, point1, point2)

avg_dst_a_img1, avg_dst_a_img2 = average_epipolar_distance(point1, lines_a_img1, point2, lines_a_img2)
avg_dst_b_img1, avg_dst_b_img2 = average_epipolar_distance(point1, lines_b_img1, point2, lines_b_img2)

# Write a_img1, a_img2, b_img1 and b_img2 into output folder.
cv2.imwrite('./output/a_img1.jpg', a_img1)
cv2.imwrite('./output/a_img2.jpg', a_img2)
cv2.imwrite('./output/b_img1.jpg', b_img1)
cv2.imwrite('./output/b_img2.jpg', b_img2)

# Output fundamental matrix in (a).
print('Fundamental matrix in (a):')
print(fundamental_matrix_a)

# Output fundamental matrix in (b).
print('Fundamental matrix in (b):')
print(fundamental_matrix_b)

# Output average distance between feature points and epipolar line in a_img1, a_img2, b_img1 and b_img2.
print("Average distance between feature points and epipolar line in a_img1:", avg_dst_a_img1)
print("Average distance between feature points and epipolar line in a_img2:", avg_dst_a_img2)
print("Average distance between feature points and epipolar line in b_img1:", avg_dst_b_img1)
print("Average distance between feature points and epipolar line in b_img2:", avg_dst_b_img2)