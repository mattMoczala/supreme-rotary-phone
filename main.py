import os
import cv2

# Load the sample image
sample = cv2.imread("./SOCOFing/Altered/Altered-Hard/70__M_Left_little_finger_OBL.BMP")

# Initialize variables for best match
best_score = 0
filename = None
image = None
kp1, kp2, mp = None, None, None

# Iterate through the first 1000 images in the Real dataset [:1000]
for file in os.listdir("./SOCOFing/Real")[:1000]:
    # Load the current fingerprint image
    fingerprint_image = cv2.imread(f"./SOCOFing/Real/{file}")

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect and compute keypoints and descriptors for the sample and current image
    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

    # Use FLANN-based matcher to find matches
    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptors_1, descriptors_2, k=2)

    # Filter good matches based on Lowe's ratio test
    match_points = [p for p, q in matches if p.distance < 0.1 * q.distance]

    # Determine the number of keypoints to calculate the score
    keypoints = min(len(keypoints_1), len(keypoints_2))

    # Calculate the matching score
    score = len(match_points) / keypoints * 100

    # Print the score for the current file
    print(f"Testing file {file} completed with score: {score}")

    # Update best match if the current score is higher
    if score > best_score:
        best_score = score
        filename = file
        image = fingerprint_image
        kp1, kp2, mp = keypoints_1, keypoints_2, match_points

# Print the best match and its score
print("BEST MATCH: " + filename)
print("SCORE: " + str(best_score))

# Draw and display the best match
result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
result = cv2.resize(result, None, fx=4, fy=4)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
