#####################################################
# Classification and Labelling Script               #
# Manually Classify New Images into Input Folders   #
#####################################################

# Imported Modules
from tkinter import *
import os
import cv2
import imutils

# Importing Movenet Models to see pose overlay
sys.path.append('TF-Movenet/examples/lite/examples/pose_estimation/raspberry_pi')
from ml import Classifier
from ml import Movenet
import utils

# Setting the starting folder path and ending folder path
# Calling the loopImages function
def main():
    FOLDER_PATH = "/Users/rahul/Desktop/TODO-Input/Front Lever"
    END_PATH = "/Users/rahul/Documents/Calisthenics-App/Input/Front Lever"
    loopImages(FOLDER_PATH, END_PATH)

# Loop through the images in selected folder
def loopImages(folder, path):
    directory = os.fsencode(folder)
    files = os.listdir(directory)

    # Loop through images in folder path
    for file in files:
        file_path = folder + "/" + os.fsdecode(file)

        if os.fsdecode(file) == ".DS_Store":
            continue

        # Obtain image and run pose estimation
        img = cv2.imread(file_path)
        pose_img = pose_estimation(img)

        # Scale up image size
        scale_factor = 1.5
        width = int(pose_img.shape[1] * scale_factor)
        height = int(pose_img.shape[0] * scale_factor)
        pose_img = cv2.resize(pose_img, (width, height))

        # Display image with pose estimation
        cv2.imshow("GUI Image Labelling", pose_img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        
        # Continuously ask the user to select X/P/G/A/B/Exit
        # This classifies each image and puts into correct folder path or deletes image or exits python script
        while True:
            action = input("X/P/G/A/B/Exit: ").lower()

            # Delete and go to Next Image
            if action == "x":
                os.remove(file_path)
                break
            
            # Put Image in good filepath
            elif action == "g":
                saveImage(file_path, path + "/Good")
                os.remove(file_path)
                break
            
            # Put Image in perfect filepath
            elif action == "p":
                saveImage(file_path, path + "/Perfect")
                os.remove(file_path)
                break

            # Put Image in average filepath
            elif action == "a":
                saveImage(file_path, path + "/Average")
                os.remove(file_path)
                break

            # Put Image in bad filepath
            elif action == "b":
                saveImage(file_path, path + "/Bad")
                os.remove(file_path)
                break

            # Exit program
            elif action == "exit":
                quit()

# Run pose estimation and return the frame
def pose_estimation(img):
    pose_detector = Movenet("TF-Movenet/movenet_thunder")
    list_persons = [pose_detector.detect(img)]
    color = (255, 255, 255)
    frame = utils.visualize(img, list_persons, color)
    return frame

# Rotating and Mirroring Image and Saving to Correct Path
def saveImage(file_path, path):
    # Count files in each directory
    count = len([x for x in os.listdir(path)])

    # Reading the image
    img = cv2.imread(file_path)

    # Rotate 2 Degrees
    rotated_left = imutils.rotate(img, angle=2)
    rotated_right = imutils.rotate(img, angle=-2)

    # Mirror Image
    mir_img = cv2.flip(img, 1)

    # Rotate Mirrored Image 2 Degrees
    mir_rotated_left = imutils.rotate(mir_img, angle=2)
    mir_rotated_right = imutils.rotate(mir_img, angle=-2)

    # Saving the Images
    cv2.imwrite(path + "/" + str(count) + ".jpeg", img)
    count+=1
    cv2.imwrite(path + "/" + str(count) + ".jpeg", rotated_left)
    count+=1
    cv2.imwrite(path + "/" + str(count) + ".jpeg", rotated_right)
    count+=1
    cv2.imwrite(path + "/" + str(count) + ".jpeg", mir_img)
    count+=1
    cv2.imwrite(path + "/" + str(count) + ".jpeg", mir_rotated_left)
    count+=1
    cv2.imwrite(path + "/" + str(count) + ".jpeg", mir_rotated_right)

if __name__ == "__main__":
    main()
