import os
import time
import cv2
import imutils
import numpy as np
#from numba import jit, njit
from imutils.video import FPS
from threading import Thread


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
        self.stopped = True


# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=3840,
    capture_height=2160,
    display_width=3840,
    display_height=2160,
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


def show_image(img, mirror=False):
    if mirror:
        img = cv2.flip(img, 1)
    cv2.imshow('my webcam', img)


def resizeImage(image, height, width):
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


#@njit(nogil=True)
def getMeanNP(img):
    img = img[..., ::-1]
    height, width, color = img.shape
    for x in range(0, height, 4):
        for y in range(0, width, 4):
            mean0 = 0
            mean1 = 0
            mean2 = 0
            area = np.copy(img[x:x+4, y:y+4])
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
                    area[j, k, 0] = mean0
                    area[j, k, 1] = mean1
                    area[j, k, 2] = mean2
            img[x:x+4, y:y+4] = area
    
    return img


#@jit(forceobj=True, parallel=True, nogil=True)
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


#@njit(nogil=True)

#def four_point_transform(image, rect):

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


# src_path = './A/'
# dst_path = './A_Ready_256/'

# def main():
#     pts1 = np.array([[119, 1190], [1986, 1130], [163, 2958], [1962, 3033]])
#     pts2 = np.array([[109, 859], [2028, 751], [218, 2619], [1971, 2685]])
#     pts3 = np.array([[44, 907], [1901, 859], [57, 2554], [1824, 2784]])
#     pts4 = np.array([[66, 1024], [1938, 963], [118, 2767], [1881, 2877]])
#     for count, filename in enumerate(os.listdir(src_path)):
#         # print(src_path + filename)
#         img = cv2.imread(src_path + filename)
#         print(img.shape)
#         start = time.time()
#         if filename[-7:-6] == "1":
#             rect = order_points(pts1)
#         elif filename[-7:-6] == "2":
#             rect = order_points(pts2)
#         elif filename[-7:-6] == "3":
#             rect = order_points(pts3)
#         else:
#             rect = order_points(pts4)

#         dst, maxWidth, maxHeight = four_point_transform(rect)
#         M = cv2.getPerspectiveTransform(rect, dst)
#         warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
#         mean = getMeanNP(warped)
#         resized = resizeImage(mean, 256, 256)
#         print('Time taken for frame {} sec\n'.format(time.time()-start))
#         #(h, w) = resized.shape[:2]
#         #centerFloat = (w / 2, h / 2)
#         #M = cv2.getRotationMatrix2D(centerFloat, -90, 1.0)
#         #rotated = cv2.warpAffine(resized, M, (w, h))
#         cv2.imwrite(dst_path + filename, warped)



def main():
    pts1 = np.array([[119, 1190], [1986, 1130], [163, 2958], [1962, 3033]])
    pts2 = np.array([[109, 859], [2028, 751], [218, 2619], [1971, 2685]])
    pts3 = np.array([[44, 907], [1901, 859], [57, 2554], [1824, 2784]])
    pts4 = np.array([[66, 1024], [1938, 963], [118, 2767], [1881, 2877]])

    vs1 = WebcamVideoStream(src=gstreamer_pipeline(sensor_id=0), device=cv2.CAP_GSTREAMER).start()
    vs2 = WebcamVideoStream(src=gstreamer_pipeline(sensor_id=1), device=cv2.CAP_GSTREAMER).start()
    vs3 = WebcamVideoStream(src=gstreamer_pipeline(sensor_id=2), device=cv2.CAP_GSTREAMER).start()
    vs4 = WebcamVideoStream(src=gstreamer_pipeline(sensor_id=3), device=cv2.CAP_GSTREAMER).start()

    rect1 = order_points(pts1)
    rect2 = order_points(pts2)
    rect3 = order_points(pts3)
    rect4 = order_points(pts4)

    dst1, maxWidth1, maxHeight1 = four_point_transform(rect1)
    dst2, maxWidth2, maxHeight2 = four_point_transform(rect2)
    dst3, maxWidth3, maxHeight3 = four_point_transform(rect3)
    dst4, maxWidth4, maxHeight4 = four_point_transform(rect4)

    M1 = cv2.getPerspectiveTransform(rect1, dst1)
    M2 = cv2.getPerspectiveTransform(rect2, dst2)
    M3 = cv2.getPerspectiveTransform(rect3, dst3)
    M4 = cv2.getPerspectiveTransform(rect4, dst4)

    print("[INFO] sampling THREADED frames from webcam...")
    fps = FPS().start()
    while True:
        frame1 = vs1.read()
        frame2 = vs2.read()
        frame3 = vs3.read()
        frame4 = vs4.read()

        warped1 = cv2.warpPerspective(frame1, M1, (maxWidth1, maxHeight1))
        warped2 = cv2.warpPerspective(frame2, M2, (maxWidth2, maxHeight2))
        warped3 = cv2.warpPerspective(frame3, M3, (maxWidth3, maxHeight3))
        warped4 = cv2.warpPerspective(frame4, M4, (maxWidth4, maxHeight4))

        cv2.imshow('frame1', warped1)
        cv2.imshow('frame2', warped2)
        cv2.imshow('frame3', warped3)
        cv2.imshow('frame4', warped4)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        fps.update()
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs1.stop()
    vs2.stop()
    vs3.stop()
    vs4.stop()


if __name__ == "__main__":
    main()
