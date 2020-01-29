# import cv2 
# import picamera 
# import picamera.array 
 
# with picamera.PiCamera() as camera: 
#     with picamera.array.PiRGBArray(camera) as stream: 
#         camera.resolution = (100, 70) 
 
#         while True: 
#             camera.capture(stream, 'bgr', use_video_port=True) 
#             # stream.array now contains the image data in BGR order 
#             image = stream.array 
#             cv2.imshow('frame', image) 
#             if cv2.waitKey(1) & 0xFF == ord('q'): 
#                 break 
#             # reset the stream before the next capture 
#             stream.seek(0) 
#             stream.truncate() 
 
#         cv2.destroyAllWindows()

import picamera

with picamera.PiCamera() as camera:
    camera.resolution = (100, 70) 
    camera.start_recording('my_video.h264')
    print("started!")
    camera.wait_recording(60)
    camera.stop_recording()  
    print("finished!")      