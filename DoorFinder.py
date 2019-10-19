'''
This software tool is the final product from my Research Experience for Undergraduates 
at Wichita State University over the summer of 2019, as funded by the National Science 
Foundation. It is based off IBeaconMap, a software tool developed by Seyed Ali Cheraghi, 
Vinod Namboodiri, and Kaushik Sinha.

The paper that outlines the specific of this software can be found here:

https://arxiv.org/abs/1802.05735

This program is able to mark and identify doors as seen on a floor plan image,
a necessary step in the deployment of bluetooth beacons in indoor navigation.
The purpose of this program is to remove the need for manual marking of doors,
such that the beacon locations for any given floor can quickly and accurately
identified.
Known bugs: As of right now, the program will crash upon exiting; however, this does not
affect or delete the output image with marked doors.

AUTHOR: Charlie Broadbent
'''


import PySimpleGUI as sg
import door_finder as df
import time
import os
import numpy as np
import cv2
import glob

sg.ChangeLookAndFeel('Reddit')

full_layout = [[sg.Text('Please enter the file of the floor plan image', auto_size_text=True)],
                [sg.Text('Floor plan image', size= (15,1), auto_size_text = False, justification='left'),
                            sg.InputText('', justification='center'), sg.FileBrowse()],
                [sg.Text('Please enter the file of the cropped door image', auto_size_text=True)],
                [sg.Text('Door image', size=(15,1), auto_size_text=False, justification='left'),
                            sg.InputText('', justification='center'), sg.FileBrowse()],
                [sg.Text(' ' * 50), sg.Submit(tooltip='Click to submit this window'), sg.Cancel()],
                [sg.Text(' ' * 10), sg.Text('', size=(50,1), key='_PROGRESS_', justification='center')]]

window = sg.Window('DoorFinder', grab_anywhere=False).Layout(full_layout).Finalize()

def find_and_match_features(floor_img, door_img):

    floor_plan = floor_img
    door_img = door_img

    img1 = cv2.imread(floor_plan) # queryImage
    img2 = cv2.imread(door_img) # floor plan Image

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create(sigma=.5)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # Use BFMatcher (Brute-Force matcher) with default params
    bf = cv2.BFMatcher()

    des1 = np.float32(des1)
    des2 = np.float32(des2)

    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test needed for SIFT
    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append(m)

    # Initialize list of match keypoints
    coord_floor = []
    coord_door = []

    for keypoint in kp2:
        (x2,y2) = keypoint.pt
        coord_door.append((x2,y2))

    # For each match...
    for mat in good:

        # Get the matching keypoints from the floor plan
        idx = mat.queryIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1,y1) = kp1[idx].pt

        # Append to each list
        coord_floor.append((x1, y1))

    coord_floor = np.float32(coord_floor) # convert 2D array to float32
    coord_door = np.float32(coord_door)

    floor_features = img1.copy()
    door_features = img2.copy()

    for pix in coord_floor:
        pixel = pix.astype(int)
        x = pixel[0]
        y = pixel[1]
        cv2.circle(floor_features, (x,y), 7, (255, 0, 0), -1)
    for pix in coord_door:
        pixel = pix.astype(int)
        x = pixel[0]
        y = pixel[1]
        cv2.circle(door_features, (x,y), 3, (255, 0, 0), -1)

    height, width, channels = img2.shape

    radius = min(width,height)

    return coord_floor, radius

def narrow_matches(coord, radius):

    clusters = 2500

    if len(coord) < 2500:
        clusters = len(coord)//10

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Cap maximum clusters to 2500 (it is unlikely any floor plan is going to have more than 2500 doors).
    if clusters > 2500:
        clusters = 2500

    # Apply KMeans-clustering. 
    compactness, labels, centers = cv2.kmeans(coord , clusters, None, criteria, 10, flags)

    centers = centers.astype(int)

    clusters_found = False # is True when no two clusters are less than 10 pixels apart from each other

    while clusters_found == False:
        reset = False
        for i in range(len(centers)-2):
            point1 = centers[i] # get coordinates of first point
            for j in range(i+1, len(centers)-1):
                point2 = centers[j] # get coordinates of second point (to check against first point)
                distance = np.linalg.norm(point1 - point2) # find distance between two points
                if distance < radius: # check if pixels are within 10 pixels of each other
                    newX = (point1[0] + point2[0])//2 # midpoint's x-coordinate
                    newY = (point1[1] + point2[1])//2 # midpoint's y-coordinate
                    midpoint = (newX, newY)
                    # delete old points in reverse order
                    if i < j:
                        centers = np.delete(centers, j, 0)
                        centers = np.delete(centers, i, 0)
                    else:
                        centers = np.delete(centers, i, 0)
                        centers = np.delete(centers, j, 0)

                    centers = np.vstack([centers, midpoint])
                    i = 0 # reset i
                    j = 0 # reset j
                    reset = True
                    break
                else:
                    j += 1
            if reset == True:
                break
            else:
                i += 1
        if reset == False:
            clusters_found = True

    return centers

def mark(coords, floor_image):

    imgDoors = cv2.imread(floor_image)

    save_file = floor_image[:-4] + '_result.jpg'
    
    # Mark each door
    for door in coords:
        pixel = door.astype(int)
        x = pixel[0]
        y = pixel[1]
        cv2.circle(imgDoors, (x, y), 10, (0, 0, 255), -1)

    cv2.imwrite(save_file, imgDoors)

    print("Image saved at: ", save_file)

    return save_file

def convert_and_scale_png(filename):
    oriimg = cv2.imread(filename)
    img_height, img_width, channels = oriimg.shape

    if img_width > 1280:
        img_height = 1280*(img_height/img_width)
        img_width = 1280
    if img_height > 720:
        img_width = 720*(img_width/img_height)
        img_height = 720

    img_width = int(img_width)
    img_height = int(img_height)

    resized_img = cv2.resize(oriimg, (img_width, img_height))
    filename_split = filename.split('/')
    filename_png = filename_split[-1][:-3] + 'png'
    cv2.imwrite(filename_png, resized_img)

    return filename_png

def main():

    while True:
        event, values = window.Read()
        if event == None:
            break
        if event== 'Cancel':
            break
        elif event == 'Submit':

            window.Element('_PROGRESS_').Update('Please wait. This may take a long time for large images.')
            window.Finalize()

            floor_plan = values[0]
            door_img = values[1]

            tic = time.perf_counter()

            floor_matches, radius = find_and_match_features(floor_plan, door_img)

            final_centers = []

            if len(floor_matches) > 10000: # large image; split into fourths to help?

                fourth = len(floor_matches)//4

                final_centers.append(narrow_matches(floor_matches[:fourth], radius))
                final_centers.append(narrow_matches(floor_matches[fourth+1:2*fourth], radius))
                final_centers.append(narrow_matches(floor_matches[2*fourth+1:3*fourth],  radius))
                final_centers.append(narrow_matches(floor_matches[3*fourth+1:], radius))
                final_centers = narrow_matches(final_centers, radius)

            else:
                final_centers = narrow_matches(floor_matches, radius)

            save_file = mark(final_centers, floor_plan)

            toc = time.perf_counter()

            processing_time = toc - tic

            print("Processing time : ", processing_time)

            floor_display = convert_and_scale_png(save_file)

            window.Element('_PROGRESS_').Update('Success! ' + floor_display + ' has been saved with the floor plan.')

            result_window = sg.Window('Result', grab_anywhere=False).Layout([[sg.Image(filename=floor_display)]]).Finalize()

            continue


main()

# Remove unnecessary images.
for f in glob.glob("*_result.png"):
    os.remove(f)

exit()


