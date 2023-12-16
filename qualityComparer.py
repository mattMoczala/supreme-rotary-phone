import os
import cv2
from statistics import mean

scores = []

def add_text_before_extension(filename, text):
    # Split the filename and extension
    name, extension = filename.rsplit('.', 1)

    # Concatenate the name, text, and extension
    new_filename = f"{name}{text}.{extension}"

    return new_filename

for file in [file for file in os.listdir("./SOCOFing/Real")][:1000]:
    print(file)
    print(add_text_before_extension(file, "_Obl"))
    
    fingerprint_image = cv2.imread("./SOCOFing/Real/" + file)
    
    path_to_altered_image = "./SOCOFing/Altered/Altered-Hard/"+ add_text_before_extension(file, "_Obl")

    if os.path.exists(path_to_altered_image):
        altered_fingerprint_image  = cv2.imread(path_to_altered_image)

        sift = cv2.SIFT_create()

        keypoints_1, descriptors_1 = sift.detectAndCompute(altered_fingerprint_image, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

        matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptors_1, descriptors_2, k=2)

        match_points = []

        for p, q in matches:
            if p.distance < 0.1 * q.distance: 
                match_points.append(p)

        keypoints = 0
        if len(keypoints_1) < len(keypoints_2):
            keypoints = len(keypoints_1)
        else:
            keypoints = len(keypoints_2)

        score = len(match_points) / keypoints * 100
        print("Testing file pair "+file+" completed with score: "+str(score))
        scores.append(score)
    else:
        print("Real fingerprint:" +file+" does not have Obl type alteration.")

print("Mean score of Obl alteration: " + str(mean(scores)))
