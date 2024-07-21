import cv2


def get_blob_detector():
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = True
    params.blobColor = 255

    params.minThreshold = 150
    params.maxThreshold = 255

    params.filterByArea = True
    params.maxArea = 1000
    
    return cv2.SimpleBlobDetector_create(params)