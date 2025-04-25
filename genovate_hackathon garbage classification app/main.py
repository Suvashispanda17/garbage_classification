import os
import cvzone
from tensorflow.keras.models import load_model  # Correct import
import cv2
from cvzone.ClassificationModule import Classifier


# Initialize video capture
cap = cv2.VideoCapture(0)

# Load the classifier model and labels
classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
imgArrow = cv2.imread('Resources/arrow.png', cv2.IMREAD_UNCHANGED)

# Initialize classIDBin
classIDBin = 0

# Import all the waste images
imgWasteList = []
pathFolderWaste = "Resources/Waste"
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

# Import all the bins images
imgBinsList = []
pathFolderBins = "Resources/Bins"
pathList = os.listdir(pathFolderBins)
for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))

# Map classID to bin types
classDic = {0: None,
            1: 0,
            2: 0,
            3: 3,
            4: 3,
            5: 1,
            6: 1,
            7: 2,
            8: 2}

while True:
    _, img = cap.read()
    imgResize = cv2.resize(img, (454, 340))

    # Load background image
    imgBackground = cv2.imread('Resources/background.png')

    # Get prediction from the classifier
    predection = classifier.getPrediction(img)

    # Extract class ID from prediction
    classID = predection[1]
    print(f"Predicted classID: {classID}")

    # Ensure classID is within the valid range for imgWasteList
    if 0 < classID <= len(imgWasteList):
        # Overlay the corresponding waste image on the background
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID - 1], (909, 127))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))

        # Get corresponding bin for the waste type
        classIDBin = classDic[classID]

        # Overlay the corresponding bin image
        if classIDBin is not None and classIDBin < len(imgBinsList):
            imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))
        else:
            print(f"Invalid bin ID: {classIDBin}, skipping bin overlay.")

    else:
        print(f"Invalid classID: {classID}, skipping waste overlay.")

    # Place the resized input image onto the background
    imgBackground[148:148 + 340, 159:159 + 454] = imgResize

    # Display the output image
    cv2.imshow("Output", imgBackground)

    # Wait for 1ms and check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release the capture object and close all windows
cap.release()
cv2.destroyAllWindows()
