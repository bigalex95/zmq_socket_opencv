from imutils.video import FPS
import numpy as np
import cv2
import os
from threading import Thread
from numba import jit, njit, vectorize, cuda, uint32, f8, uint8


class WebcamVideoStream:
    def __init__(self, src=0, device=None):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src, device)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
    	    # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
	    
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=3840,
    capture_height=2160,
    display_width=2560,
    display_height=1440,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


@cuda.jit
def getMeanNP(img, mean):
    img = img[..., ::-1]
    height = img.shape[0]
    width = img.shape[1]
    k, j = cuda.grid(2)
    for x in range(0, height, 4):
        for y in range(0, width, 4):
            area = img[x:x+4, y:y+4]
            mean0 = 0
            mean1 = 0
            mean2 = 0
            for j in range(4):
                for k in range(4):
                    mean0 += area[j, k, 0]
                    mean1 += area[j, k, 1]
                    mean2 += area[j, k, 2]
            mean0 = mean0 / 16
            mean1 = mean1 / 16
            mean2 = mean2 / 16
            for j in range(4):
                for k in range(4):
                    img[x+j, y+k, 0] = mean0
                    img[x+j, y+k, 1] = mean1
                    img[x+j, y+k, 2] = mean2
    mean = img

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(rect):
    # obtain a consistent order of the points and unpack them
    # individually
    
    #rect = order_points(pts)
    
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
	[0, 0],
	[maxWidth - 1, 0],
	[maxWidth - 1, maxHeight - 1],
    	[0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    
    #M = cv2.getPerspectiveTransform(rect, dst)
    #warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # return the warped image
    
    #return warped
    
    return dst, maxWidth, maxHeight

def main():
    pts = np.array([[85, 520], [1330, 520], [80, 1680], [1245, 1775]])
    vs1 = WebcamVideoStream(src=gstreamer_pipeline(sensor_id=2), device=cv2.CAP_GSTREAMER).start()
    
    #img = cv2.imread("frame16.jpg")
    #image_np = img[520:520+1200, 90:90+1200]

    rect = order_points(pts)
    dst, maxWidth, maxHeight = four_point_transform(rect)
    M = cv2.getPerspectiveTransform(rect, dst)
    fps = FPS().start()
    stream = cuda.stream()

    while True:
        try:
            frame1 = vs1.read()
            frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image_np = cv2.warpPerspective(frame1, M, (maxWidth, maxHeight))
            d_image = cuda.to_device(image_np, stream=stream)
            mean_np = np.empty_like(image_np)
            d_mean = cuda.to_device(mean_np, stream=stream)
            getMeanNP(d_image, d_mean)
            print(d_mean)
            tmp_np = np.asarray(d_mean)
            resize = cv2.resize(tmp_np, (256, 256), interpolation = cv2.INTER_AREA)
            cv2.imshow("mean.jpg", resize)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                cv2.imwrite('resize.jpg', resize)
            fps.update()
        except Exception as e:
            vs1.stop()
            print(e)
            break
    
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs1.stop()

if __name__ == "__main__":
    main()
