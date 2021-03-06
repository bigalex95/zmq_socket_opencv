from imutils.video import FPS
import numpy as np
import cv2
import imagiz


def main():
    server=imagiz.Server()

    fps = FPS().start()
    while True:
        try:
            message=server.recive()
            frame = cv2.imdecode(message.image,1)
            cv2.imshow("frame",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps.update()
        except Exception as e:
            print(e)
            cv2.destroyAllWindows()

            break
    
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
