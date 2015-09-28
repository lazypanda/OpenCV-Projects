import cv2  # OpenCV Library
import random # random number generator
 
#-----------------------------------------------------------------------------
#       Load and configure Haar Cascade Classifiers
#-----------------------------------------------------------------------------
 
# location of OpenCV Haar Cascade Classifiers:
baseCascadePath = '/usr/local/share/OpenCV/haarcascades/'
 
# xml files describing our haar cascade classifiers
faceCascadeFilePath = baseCascadePath + 'haarcascade_frontalface_default.xml'
noseCascadeFilePath = baseCascadePath + 'haarcascade_mcs_nose.xml'
 
# build our cv2 Cascade Classifiers
faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
noseCascade = cv2.CascadeClassifier(noseCascadeFilePath)
 
#-----------------------------------------------------------------------------
#       Load and configure replacement (.png with alpha transparency)
#-----------------------------------------------------------------------------
 
# Load our overlay image: replacement.png
imgReplacements = [cv2.imread('1.png',-1),
                   cv2.imread('2.png',-1),
                   cv2.imread('3.png',-1),
                   cv2.imread('4.png', -1),
                   cv2.imread('5.png', -1),
                   cv2.imread('6.png', -1),]

def getImageReplacementData():
    index = random.randint(0,len(imgReplacements)-1)
    index = 1

    imgReplacement = imgReplacements[index]

    # Create the mask for the replacement
    orig_mask = imgReplacement[:,:,3]
     
    # Create the inverted mask for the replacement
    orig_mask_inv = cv2.bitwise_not(orig_mask)
     
    # Convert replacement image to BGR
    # and save the original image size (used later when re-sizing the image)
    imgReplacement = imgReplacement[:,:,0:3]
    origReplacementHeight, origReplacementWidth = imgReplacement.shape[:2]

    return orig_mask, orig_mask_inv, imgReplacement, origReplacementHeight, origReplacementWidth
 
#-----------------------------------------------------------------------------
#       Main program loop
#-----------------------------------------------------------------------------
 
# collect video input from first webcam on system
video_capture = cv2.VideoCapture(0)
 
while True:
    # Capture video feed
    ret, frame = video_capture.read()
 
    # Create greyscale image from the video feed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # Detect faces in input video stream
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
 
   # Iterate over each face found
    for (x, y, w, h) in faces:
        # Un-comment the next line for debug (draw box around all faces)
        face = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
 
        # Detect a nose within the region bounded by each face (the ROI)
        nose = noseCascade.detectMultiScale(roi_gray)
 
        for (nx,ny,nw,nh) in nose:
            orig_mask, orig_mask_inv, imgReplacement, origReplacementHeight, origReplacementWidth = getImageReplacementData()
            # Un-comment the next line for debug (draw box around the nose)
            cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),2)
 
            # The replacement should be three times the width of the nose
            replacementWidth =  h
            replacementHeight = replacementWidth * origReplacementHeight / origReplacementWidth
 
            # Center the replacement on the bottom of the nose
            x1 = nx - (replacementWidth)
            x2 = nx + nw + (replacementWidth)
            y1 = ny - (replacementHeight)
            y2 = ny + nh + (replacementHeight)
 
            # Check for clipping
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h

            cv2.rectangle(roi_color,(x1,y1),(x2,y2),(0,0,255),2)
 
            # Re-calculate the width and height of the replacement image
            replacementWidth = 2 * (x2 - x1)
            replacementHeight = 2 * (y2 - y1)
 
            # Re-size the original image and the masks to the replacement sizes
            # calculated above
            replacement = cv2.resize(imgReplacement, (replacementWidth,replacementHeight), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (replacementWidth,replacementHeight), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (replacementWidth,replacementHeight), interpolation = cv2.INTER_AREA)
 
            # take ROI for replacement from background equal to size of replacement image
            roi = cv2.medianBlur(roi_color[y1:y2, x1:x2], 15, 0)
 
            # roi_bg contains the original image only where the replacement is not
            # in the region that is the size of the replacement.
            roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
 
            # roi_fg contains the image of the replacement only where the replacement is
            roi_fg = cv2.bitwise_and(replacement,replacement,mask = mask)
 
            # join the roi_bg and roi_fg
            dst = cv2.add(roi_bg,roi_fg)
 
            # place the joined image, saved to dst back over the original image
            roi_color[y1:y2, x1:x2] = dst
 
            break
 
    # Display the resulting frame
    cv2.imshow('Video', frame)
 
    # press any key to exit
    # NOTE;  x86 systems may need to remove: '& 0xFF == ord('q')'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

