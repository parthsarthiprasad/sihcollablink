import os
from sys import platform as _platform
import yaml

import numpy as np
import cv2

from afy.videocaptureasync import VideoCaptureAsync
from afy.arguments import opt
from afy.utils import info, Tee, crop, pad_img, resize, TicToc
import afy.camera_selector as cam_selector
from Local_Predictor import Predictor

log = Tee('./var/log/cam_fomm.log')


if _platform == 'darwin':
    if not opt.is_client:
        info('\nOnly remote GPU mode is supported for Mac (use --is-client and --connect options to connect to the server)')
        info('Standalone version will be available lately!\n')
        exit()


def print_help():
    info('\n\n=== Control keys ===')
    info('W: Zoom camera in')
    info('S: Zoom camera out')
    info('I: Show FPS')
    info('ESC: Quit')
    info('\n\n')


def draw_fps(frame, fps, timing, x0=10, y0=20, ystep=30, fontsz=0.5, color=(255, 255, 255)):
    frame = frame.copy()
    cv2.putText(frame, f"FPS: {fps:.1f}", (x0, y0 + ystep * 0), 0, fontsz * IMG_SIZE / 256, color, 1)
    cv2.putText(frame, f"Model time (ms): {timing['predict']:.1f}", (x0, y0 + ystep * 1), 0, fontsz * IMG_SIZE / 256, color, 1)
    cv2.putText(frame, f"Preproc time (ms): {timing['preproc']:.1f}", (x0, y0 + ystep * 2), 0, fontsz * IMG_SIZE / 256, color, 1)
    cv2.putText(frame, f"Postproc time (ms): {timing['postproc']:.1f}", (x0, y0 + ystep * 3), 0, fontsz * IMG_SIZE / 256, color, 1)
    return frame


def draw_calib_text(frame, thk=2, fontsz=0.5, color=(0, 0, 255)):
    frame = frame.copy()
    cv2.putText(frame, "FIT FACE IN RECTANGLE", (40, 20), 0, fontsz * IMG_SIZE / 255, color, thk)
    cv2.putText(frame, "W - ZOOM IN", (60, 40), 0, fontsz * IMG_SIZE / 255, color, thk)
    cv2.putText(frame, "S - ZOOM OUT", (60, 60), 0, fontsz * IMG_SIZE / 255, color, thk)
    cv2.putText(frame, "THEN PRESS X", (60, 245), 0, fontsz * IMG_SIZE / 255, color, thk)
    return frame


def select_camera(config):
    cam_config = config['cam_config']
    cam_id = None

    if os.path.isfile(cam_config):
        with open(cam_config, 'r') as f:
            cam_config = yaml.load(f, Loader=yaml.FullLoader)
            cam_id = cam_config['cam_id']
    else:
        cam_frames = cam_selector.query_cameras(config['query_n_cams'])

        if cam_frames:
            cam_id = cam_selector.select_camera(cam_frames, window="CLICK ON YOUR CAMERA")
            log(f"Selected camera {cam_id}")

            with open(cam_config, 'w') as f:
                yaml.dump({'cam_id': cam_id}, f)
        else:
            log("No cameras are available")

    return cam_id


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    global display_string
    display_string = ""

    IMG_SIZE = 256

    log('Loading Predictor')
    model_dir = "F:/Projects/Beast/Models/Inpainting/Functional_Mask_Trial_3.h5"
    
    predictor = Predictor(model_dir,"F:/Models/shape_predictor_68_face_landmarks.dat")

    cam_id = select_camera(config)

    if cam_id is None:
        exit(1)

    cap = VideoCaptureAsync(cam_id)
    cap.start()

    enable_vcam = not opt.no_stream

    ret, frame = cap.read()
    stream_img_size = frame.shape[1], frame.shape[0]
    input_mask = np.ones((128,128,3))

    if enable_vcam:
        if _platform in ['linux', 'linux2']:
            try:
                import pyfakewebcam
            except ImportError:
                log("pyfakewebcam is not installed.")
                exit(1)

            stream = pyfakewebcam.FakeWebcam(f'/dev/video{opt.virt_cam}', *stream_img_size)
        else:
            enable_vcam = False
            # log("Virtual camera is supported only on Linux.")
        
        # if not enable_vcam:
            # log("Virtual camera streaming will be disabled.")

    passthrough = False

    cv2.namedWindow('cam', cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow('cam', 500, 250)

    frame_proportion = 0.9
    frame_offset_x = 0
    frame_offset_y = 0

    overlay_alpha = 0.0
    preview_flip = False
    output_flip = False
    find_keyframe = False
    is_calibrated = False

    fps_hist = []
    fps = 0
    show_fps = False

    print_help()

    try:
        while True:
            tt = TicToc()

            timing = {
                'preproc': 0,
                'predict': 0,
                'postproc': 0
            }

            green_overlay = False

            tt.tic()

            ret, frame = cap.read()
            if not ret:
                log("Can't receive frame (stream end?). Exiting ...")
                break

            frame = frame[..., ::-1]
            frame_orig = frame.copy()

            frame, lrudwh = crop(frame, p=frame_proportion, offset_x=frame_offset_x, offset_y=frame_offset_y)
            frame_lrudwh = lrudwh
            frame = resize(frame, (IMG_SIZE, IMG_SIZE))[..., :3]


            timing['preproc'] = tt.toc()

            if passthrough:
                out = frame
            elif is_calibrated:
                tt.tic()
                out = frame.copy()
                coords = predictor.extract_landmarks(out)
                if(type(coords)!=int): 
                    face_coords,eye_marks = coords
                    input_img = cv2.resize(out[face_coords[2]:face_coords[3],face_coords[0]:face_coords[1]],(128,128))/255
                    cv2.fillConvexPoly(input_mask,np.array(eye_marks[0:6]),(0,0,0))
                    cv2.fillConvexPoly(input_mask,np.array(eye_marks[6:12]),(0,0,0))
                    pred_img = predictor.predict(input_img,input_mask)
                    out[face_coords[2]:face_coords[3],face_coords[0]:face_coords[1]] = cv2.resize(pred_img,(face_coords[1]-face_coords[0],face_coords[3]-face_coords[2]))*255
                else:
                    log("Faces Not found.")
                timing['predict'] = tt.toc()
            else:
                out = None

            tt.tic()
            
            key = cv2.waitKey(1)

            if key == 27: # ESC
                break
            elif key == ord('w'):
                frame_proportion -= 0.05
                frame_proportion = max(frame_proportion, 0.1)
            elif key == ord('s'):
                frame_proportion += 0.05
                frame_proportion = min(frame_proportion, 1.0)
            elif key == ord('H'):
                if frame_lrudwh[0] - 1 > 0:
                    frame_offset_x -= 1
            elif key == ord('h'):
                if frame_lrudwh[0] - 5 > 0:
                    frame_offset_x -= 5
            elif key == ord('K'):
                if frame_lrudwh[1] + 1 < frame_lrudwh[4]:
                    frame_offset_x += 1
            elif key == ord('k'):
                if frame_lrudwh[1] + 5 < frame_lrudwh[4]:
                    frame_offset_x += 5
            elif key == ord('J'):
                if frame_lrudwh[2] - 1 > 0:
                    frame_offset_y -= 1
            elif key == ord('j'):
                if frame_lrudwh[2] - 5 > 0:
                    frame_offset_y -= 5
            elif key == ord('U'):
                if frame_lrudwh[3] + 1 < frame_lrudwh[5]:
                    frame_offset_y += 1
            elif key == ord('u'):
                if frame_lrudwh[3] + 5 < frame_lrudwh[5]:
                    frame_offset_y += 5
            elif key == ord('Z'):
                frame_offset_x = 0
                frame_offset_y = 0
                frame_proportion = 0.9
            elif key == ord('x'):
                predictor.reset_frames()

                if not is_calibrated:
                    cv2.namedWindow('avatarify', cv2.WINDOW_GUI_NORMAL)
                    cv2.moveWindow('avatarify', 600, 250)
                
                is_calibrated = True
            elif key == ord('z'):
                overlay_alpha = max(overlay_alpha - 0.1, 0.0)
            elif key == ord('c'):
                overlay_alpha = min(overlay_alpha + 0.1, 1.0)
            elif key == ord('r'):
                preview_flip = not preview_flip
            elif key == ord('t'):
                output_flip = not output_flip
            elif key == ord('f'):
                find_keyframe = not find_keyframe
            elif key == ord('i'):
                show_fps = not show_fps
            elif key == ord('s'):
                passthrough = not passthrough
            elif key != -1:
                log(key)

            timing['postproc'] = tt.toc()
                

            cv2.imshow('cam', frame[..., ::-1])

            if out is not None:
                if not opt.no_pad:
                    out = pad_img(out, stream_img_size)

                if output_flip:
                    out = cv2.flip(out, 1)

                if enable_vcam:
                    out = resize(out, stream_img_size)
                    stream.schedule_frame(out)

                cv2.imshow('avatarify', out[..., ::-1])

            fps_hist.append(tt.toc(total=True))
            if len(fps_hist) == 10:
                fps = 10 / (sum(fps_hist) / 1000)
                fps_hist = []
    except KeyboardInterrupt:
        log("main: user interrupt")

    log("stopping camera")
    cap.stop()

    cv2.destroyAllWindows()

    if opt.is_client:
        log("stopping remote predictor")
        predictor.stop()

    log("main: exit")
