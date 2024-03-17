import cv2
import numpy as np

# A. SIFT interest point detection
def interesting_point(image, cthreshold, ethreshold):
    ## a. Apply SIFT interesting point detector to 1a_notredame image and 1b_notredame image.
    ## b. Adjust the related thresholds in SIFT detection such that there are between 100 interest points and 500 interest points detected in 1a_notredame image and 1b_notredame image.
    sift = cv2.SIFT_create(contrastThreshold = cthreshold, edgeThreshold = ethreshold) 
    # detect keypoints and compute their descriptors.
    keypoint, descriptor = sift.detectAndCompute(image, None)
    ## c. Plot the detected interest points on the corresponding image.
    image = cv2.drawKeypoints(image, keypoint, image)
    # return image with keypoints, keypoints and descriptors.
    return image, keypoint, descriptor

# B. SIFT feature matching
## b. Implement a function that finds a list of interest point correspondences based on nearest-neighbor matching principle.
def feature_matching(img1_features, img2_features, dthreshold):
    matches = []
    for i in range(img1_features.shape[0]):
        ## a. Compare the similarity between all the pairs between the detected interest points from each of the two images based on a suitable distance function between two SIFT feature vectors.
        distances = np.sqrt(np.square(img1_features[i,:]-img2_features).sum(axis = 1)) # euclidean distance
        index_sorted = np.argsort(distances) # sort distance in ascending order
        if(distances[index_sorted[0]] < dthreshold * distances[index_sorted[1]]):
            matches.append([i, index_sorted[0]])
    # return coordinate that two pictures match.
    return np.array(matches)

# Read 1a_notredame image and 1b_notredame image.
img1 = cv2.imread('1a_notredame.jpg')
img2 = cv2.imread('1b_notredame.jpg')

img1, keypoint1, descriptor1 = interesting_point(img1, cthreshold = 0.2, ethreshold = 20)
img2, keypoint2, descriptor2 = interesting_point(img2, cthreshold = 0.2, ethreshold = 10)

# Write 1a_notredame with keypoints image and 1b_notredame with keypoints image into output folder.
cv2.imwrite('./output/1a_notredame_SIFT_interest_point_detection.jpg', img1)
cv2.imwrite('./output/1b_notredame_SIFT_interest_point_detection.jpg', img2)

Dmatch = feature_matching(descriptor1, descriptor2, 0.7)
match = [cv2.DMatch(Dmatch[i][0], Dmatch[i][1], 0) for i in range(Dmatch.shape[0])]
## c. Plot the point corresponces overlaid on the pair of original images.
img_match = cv2.drawMatches(img1, keypoint1, img2, keypoint2, match, None)

# Write matching 1a_notredame features and 1b_notredame features image into output folder.
cv2.imwrite('./output/notredame_SIFT_feature_matching.jpg', img_match)