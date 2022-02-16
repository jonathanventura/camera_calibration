import cv2
from cv2 import aruco

def calibrateCamera(pathToVideo):
    row_count = 7
    col_count = 5
    num_markers_required = 5
    # specifies type of aruco markers to look for
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
    charuco_board = aruco.CharucoBoard_create(
        squaresX=col_count,
        squaresY=row_count,
        squareLength=0.04,
        markerLength=0.02,
        dictionary=aruco_dict
    )

    all_corners = []
    all_ids = []
    images_found = 0
    image_size = None

    cap = cv2.VideoCapture(pathToVideo)
    while(True):
        is_image, frame = cap.read()
        if(not is_image):
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detects aruco Markers
        # aruco_dict specifies markers to look for
        corners, ids, _ = aruco.detectMarkers(
            image=gray,
            dictionary=aruco_dict
        )

        img = aruco.drawDetectedMarkers(
            image=frame,
            corners=corners
        )

        # Displays image with detected markers.
        #cv2.imshow('display', img)
        # cv2.waitKey(0)

        # Used to check if any markers/corners found in given image.
        if(len(corners) == 0):
            continue

        # interpolates aruco markers found previously
        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=charuco_board
        )

        # check to see if response has atleast as many markers as needed for processing
        if response >= num_markers_required:
            images_found += 1
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)

            if image_size is None:
                image_size = gray.shape[::-1]
        else:
            print('Not able to either detect board or not enough markers detected')
    cv2.destroyAllWindows()

    if(images_found == 0):
        return
      
    # gives calibration data based on markers and ids previously found.
    calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=charuco_board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    print(cameraMatrix)
    print(distCoeffs)
    print(rvecs)
    print(tvecs)
    return


if __name__ == "__main__":
    pathToVideo = None
    calibrateCamera(pathToVideo)
