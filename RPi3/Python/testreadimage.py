# import io
# import time
# import picamera
# import cv2
# import numpy as np


# if __name__ == "__main__":

#     try: 
#         while (1): 
#             with picamera.PiCamera() as camera:
#                 stream = io.BytesIO()

#                 camera.capture(stream, format='jpeg')
#                 data = np.fromstring(stream.getvalue(), dtype=np.uint8)
#                 image = cv2.imdecode(data, 1)

#                 # image = image[:, :, ::-1]
#                 cv2.imshow('frame', image)
#                 time.sleep(0.1)


#     except KeyboardInterrupt:
#         camera.close()
#         print('KeyboardInterrupt')

import cv2
import picamera
import picamera.array

with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as stream:
        camera.resolution = (100, 70)

        while True:
            camera.capture(stream, 'bgr', use_video_port=True)
            # stream.array now contains the image data in BGR order
            image = stream.array
            cv2.imshow('frame', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # reset the stream before the next capture
            stream.seek(0)
            stream.truncate()

        cv2.destroyAllWindows()