import cv2
import numpy as np
import random

# A. For SIFT interest point detection to extract the SIFT feature points from images.
def interesting_point(image, cthreshold, ethreshold):
    # Apply SIFT interesting point detector to image.
    # Adjust the related thresholds in SIFT detection.
    sift = cv2.SIFT_create(contrastThreshold = cthreshold, edgeThreshold = ethreshold) 
    # Detect keypoints and compute their descriptors.
    keypoint, descriptor = sift.detectAndCompute(image, None)
    # Plot the detected interest points on the corresponding image.
    image = cv2.drawKeypoints(image, keypoint, image)
    # Return image with keypoints, keypoints and descriptors.
    return image, keypoint, descriptor

# A. Establish point correspondences between the SIFT feature points detected from the single-book images and the cluttered-book image.
def feature_matching(img1_features, img2_features, dthreshold):
    matches = []
    for i in range(img1_features.shape[0]):
        # A. Using the distance between the SIFT feature vectors as a matching score
        distances = np.sqrt(np.square(img1_features[i,:]-img2_features).sum(axis = 1)) 
        index_sorted = np.argsort(distances) # Sort distance in ascending order.
        if(distances[index_sorted[0]] < dthreshold * distances[index_sorted[1]]):
            matches.append([i, index_sorted[0]])
    # Return coordinate that two pictures match.
    return np.array(matches)

# Implement a function that estimates the homography matrix H that maps a set of interest points to a new set of interest points.
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
    try:
        h = np.dot(np.linalg.inv(A), b) # Solve Ah = b by multiply A^-1 to both side.
        H = np.array([[h[0][0], h[1][0], h[2][0]],
                      [h[3][0], h[4][0], h[5][0]],
                      [h[6][0], h[7][0], 1]
                    ])
        # Return homography matrix.            
        return H
    except:
        pass

# B. Apply the RANSAC program
def RANSAC(image, match, kp1, kp2, max_iters, epsilon):
    h, w = image.shape[0], image.shape[1]
    image_copy = np.copy(image)
    best_matches = []
    for i in range(max_iters):
        # Select four feature pairs (at random).
        s = random.sample(range(len(match)), 4)
        src = [kp1[match[s[0]].queryIdx].pt, kp1[match[s[1]].queryIdx].pt, kp1[match[s[2]].queryIdx].pt, kp1[match[s[3]].queryIdx].pt]
        dst = [kp2[match[s[0]].trainIdx].pt, kp2[match[s[1]].trainIdx].pt, kp2[match[s[2]].trainIdx].pt, kp2[match[s[3]].trainIdx].pt]
        # Compute the homography matrix H (exact)
        H = homography(src, dst)
        # Compute and save inliers where SSD(p', Hp) < epsilon 
        inliers = []
        for i in range(len(match)):
            try:
                x = kp1[match[i].queryIdx].pt[0]
                y = kp1[match[i].queryIdx].pt[1]
                new_pos = np.dot(H, np.array([[x, y, 1]]).T)
                SSD = np.sqrt(np.sum(np.square(kp2[match[i].trainIdx].pt - np.array([new_pos[0][0] / new_pos[2][0], new_pos[1][0] / new_pos[2][0]]))))
                if SSD < epsilon:
                    # B. Showing the deviation vectors between the transformed feature points and the corresponding feature points on the input image.
                    image = cv2.arrowedLine(image, (int(kp2[match[i].trainIdx].pt[0]), int(kp2[match[i].trainIdx].pt[1])), (int(new_pos[0][0] / new_pos[2][0]), int(new_pos[1][0] / new_pos[2][0])), (0, 255, 0), 2)
                    inliers.append(match[i])
            except:
                pass
        # Keep the largest set of inliers
        if len(inliers) > len(best_matches):
            best_matches = inliers
            best_H = H
    # Find point of original book in 1-image after tranformed by homography matrix.
    new_pts1 = np.dot(best_H, np.array([[0, 0, 1]]).T)
    new_pts2 = np.dot(best_H, np.array([[0, h-1, 1]]).T)
    new_pts3 = np.dot(best_H, np.array([[w-1, h-1, 1]]).T)
    new_pts4 = np.dot(best_H, np.array([[w-1, 0, 1]]).T)
    new_x1, new_y1 = new_pts1[0][0] / new_pts1[2][0], new_pts1[1][0] / new_pts1[2][0]
    new_x2, new_y2 = new_pts2[0][0] / new_pts2[2][0], new_pts2[1][0] / new_pts2[2][0]
    new_x3, new_y3 = new_pts3[0][0] / new_pts3[2][0], new_pts3[1][0] / new_pts3[2][0]
    new_x4, new_y4 = new_pts4[0][0] / new_pts4[2][0], new_pts4[1][0] / new_pts4[2][0]
    # Draw area of original book that have been transformed by homography matrix.
    image_copy = cv2.line(image_copy, (int(new_x1), int(new_y1)), (int(new_x2), int(new_y2)), (0, 255, 0), 4) # top line
    image_copy = cv2.line(image_copy, (int(new_x2), int(new_y2)), (int(new_x3), int(new_y3)), (0, 255, 0), 4) # right line
    image_copy = cv2.line(image_copy, (int(new_x1), int(new_y1)), (int(new_x4), int(new_y4)), (0, 255, 0), 4) # left line
    image_copy = cv2.line(image_copy, (int(new_x4), int(new_y4)), (int(new_x3), int(new_y3)), (0, 255, 0), 4) # bottom line
    # Return inlier match point and 1-image with area of original book that have been transformed by homography matrix.
    return best_matches, image_copy


# Read 1-book1 image, 1-book2 image, 1-book3 image and 1-image image.
img1 = cv2.imread('1-book1.jpg')
img2 = cv2.imread('1-book2.jpg')
img3 = cv2.imread('1-book3.jpg')
img4 = cv2.imread('1-image.jpg')
img5 = cv2.imread('1-image.jpg')

# The number of correspondences is larger than 500
img1, keypoint1, descriptor1 = interesting_point(img1, cthreshold = 0.01, ethreshold = 10)
img2, keypoint2, descriptor2 = interesting_point(img2, cthreshold = 0.01, ethreshold = 5)
img3, keypoint3, descriptor3 = interesting_point(img3, cthreshold = 0.01, ethreshold = 10)
img4, keypoint4, descriptor4 = interesting_point(img4, cthreshold = 0.005, ethreshold = 30)

Dmatch1 = feature_matching(descriptor1, descriptor4, 0.6)
Dmatch2 = feature_matching(descriptor2, descriptor4, 0.6)
Dmatch3 = feature_matching(descriptor3, descriptor4, 0.6)
match1 = [cv2.DMatch(Dmatch1[i][0], Dmatch1[i][1], 0) for i in range(Dmatch1.shape[0])]
match2 = [cv2.DMatch(Dmatch2[i][0], Dmatch2[i][1], 0) for i in range(Dmatch2.shape[0])]
match3 = [cv2.DMatch(Dmatch3[i][0], Dmatch3[i][1], 0) for i in range(Dmatch3.shape[0])]

# A.
img1_match = cv2.drawMatches(img1, keypoint1, img4, keypoint4, match1, None)
img2_match = cv2.drawMatches(img2, keypoint2, img4, keypoint4, match2, None)
img3_match = cv2.drawMatches(img3, keypoint3, img4, keypoint4, match3, None)
cv2.imwrite('./output/Matching 1-book1 and 1-image.jpg', img1_match)
cv2.imwrite('./output/Matching 1-book2 and 1-image.jpg', img2_match)
cv2.imwrite('./output/Matching 1-book3 and 1-image.jpg', img3_match)

# B.
match1_ransac, img6 = RANSAC(img5, match1, keypoint1, keypoint4, max_iters = 100, epsilon = 1)
match2_ransac, img7 = RANSAC(img5, match2, keypoint2, keypoint4, max_iters = 100, epsilon = 1)
match3_ransac, img8 = RANSAC(img5, match3, keypoint3, keypoint4, max_iters = 100, epsilon = 1)
img1_match_ransac = cv2.drawMatches(img1, keypoint1, img6, keypoint4, match1_ransac, None)
img2_match_ransac = cv2.drawMatches(img2, keypoint2, img7, keypoint4, match2_ransac, None)
img3_match_ransac = cv2.drawMatches(img3, keypoint3, img8, keypoint4, match3_ransac, None)
cv2.imwrite('./output/Matching 1-book1 and 1-image with RANSAC.jpg', img1_match_ransac)
cv2.imwrite('./output/Matching 1-book2 and 1-image with RANSAC.jpg', img2_match_ransac)
cv2.imwrite('./output/Matching 1-book3 and 1-image with RANSAC.jpg', img3_match_ransac)
cv2.imwrite('./output/1-image with deviation vectors.jpg', img5)