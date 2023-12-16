import os
import cv2
from statistics import mean

def addTextBeforeExtension(filename: str, text: str):
    # Split the filename and extension
    name, extension = filename.rsplit('.', 1)

    # Concatenate the name, text, and extension
    newFilename = f"{name}{text}.{extension}"

    return newFilename

def printScoresForAlteration(alterationLevel: str, alterationType: str, printLogs: bool):
    scores = []
    scoresMale = []
    scoresFemale = []

    # Iterate through the first 1000 images in the Real dataset [:1000]
    for file in [file for file in os.listdir("./SOCOFing/Real")]:    
        fingerprintImage = cv2.imread("./SOCOFing/Real/" + file)
        
        pathToAlteredImage = "./SOCOFing/Altered/"+alterationLevel+"/"+ addTextBeforeExtension(file, "_"+alterationType)

        if os.path.exists(pathToAlteredImage):
            alteredFingerprintImage  = cv2.imread(pathToAlteredImage)

            sift = cv2.SIFT_create()

            keypoints1, descriptors1 = sift.detectAndCompute(alteredFingerprintImage, None)
            keypoints2, descriptors2 = sift.detectAndCompute(fingerprintImage, None)

            matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptors1, descriptors2, k=2)

            matchPoints = []

            for p, q in matches:
                if p.distance < 0.1 * q.distance: 
                    matchPoints.append(p)

            keypoints = 0
            if len(keypoints1) < len(keypoints2):
                keypoints = len(keypoints1)
            else:
                keypoints = len(keypoints2)

            score = len(matchPoints) / keypoints * 100
            scores.append(score)
            if 'M' in file and ('F' not in file or file.find('M') < file.find('F')):
                scoresMale.append(score)
            elif 'F' in file and ('M' not in file or file.find('F') < file.find('M')):
                scoresFemale.append(score)

            if printLogs:
                print("Testing file pair "+file+" completed with general score: "+str(score))
        else:
            if printLogs:
                print("Real fingerprint:" +file+" does not have Obl type alteration.")

    print("General mean score of "+alterationLevel+" "+alterationType+": " + str(mean(scores)))
    print("Female mean score of "+alterationLevel+" "+alterationType+": " + str(mean(scoresFemale)))
    print("Male mean score of "+alterationLevel+" "+alterationType+": " + str(mean(scoresMale)))
    print("\n\n")

def printAllAlterations():
    printScoresForAlteration("Altered-Hard", "Obl", False)
    printScoresForAlteration("Altered-Medium", "Obl", False)
    printScoresForAlteration("Altered-Easy", "Obl", False)

    printScoresForAlteration("Altered-Hard", "Cr", False)
    printScoresForAlteration("Altered-Medium", "Cr", False)
    printScoresForAlteration("Altered-Easy", "Cr", False)

    printScoresForAlteration("Altered-Hard", "Zcut", False)
    printScoresForAlteration("Altered-Medium", "Zcut", False)
    printScoresForAlteration("Altered-Easy", "Zcut", False)


print("\n")
printAllAlterations()
