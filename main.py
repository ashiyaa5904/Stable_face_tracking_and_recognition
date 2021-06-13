import time
from djitellopy import tello
from face_recognition.detect import get_encode, get_face, load_pickle
import cv2
import mtcnn
from scipy.spatial.distance import cosine
from face_recognition.architecture import *
from face_recognition.train_data import l2_normalizer
from Stable_Drone_Tracking.Stable_Face_Recognition import trackface
from face_tracking_with_deepsort import DeepsortFaceTracker

drone = tello.Tello()
drone.connect()
print(drone.get_battery())

drone.streamon()
drone.takeoff()
drone.send_rc_control(0, 0, 15, 0)
time.sleep(2.2)


# cap = cv2.VideoCapture(0)


def click_on_face(event, x, y, flags, params):
    # checking for left mouse DBL clicks
    global Track, TrackId
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # print(x, ' ', y,params)
        p1, p2, p3, p4 = params[0]
        # checking face
        if (x > p1) and (x < p3) and (y > p2) and (y < p4):
            print(f"track {params[1]}")
            Track = True
            TrackId = params[1]


class PersonTracker:
    def __init__(self):
        self.confidence_t = 0.99
        self.recognition_t = 0.5
        self.required_size = (160, 160)

    def person_tracker(self, img, detector, encoder, encoding_dict, face_deepsort, deepsort):
        global info
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # detect faces
        results = detector.detect_faces(img_rgb)

        boxes, class_label, confs, f = [], [], [], 0
        for res in results:
            if res['confidence'] < self.confidence_t:
                continue
            face, pt_1, pt_2 = get_face(img_rgb, res['box'])
            encode = get_encode(encoder, face, self.required_size)
            encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
            name = 'unknown'
            f = f + 1
            distance = float("inf")
            for db_name, db_encode in encoding_dict.items():
                dist = cosine(db_encode, encode)
                if dist < self.recognition_t and dist < distance:
                    name = db_name
                    distance = dist

            boxes.append(res['box'])
            class_label.append(name)
            confs.append(res["confidence"])
            # print(boxes,class_label,confs)
        if len(boxes) > 0:
            # start tracking with deepsort
            outputs = face_deepsort.start_tracking(deepsort, boxes, confs, img)
            # fps = detector.calculate_fps(start_time, f)
            # print(outputs)
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                for i, box in enumerate(bbox_xyxy):
                    color = (0, 255, 0)
                    id = int(identities[i])
                    name = class_label[::-1][i]
                    # label_c = f"{label}{id} {confs[i]:.2f}"
                    x1, y1, x2, y2 = box
                    # print(box)
                    if name == 'unknown':
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img, name + str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                    else:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, name + str(id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 200, 200), 2)

                        # start tracking the person
                    if Track:
                        if TrackId == id:
                            pw = x2 - x1
                            ph = y2 - y1
                            cx = x1 + pw // 2
                            cy = y1 + ph // 2
                            area = pw * ph
                            info = [[cx, cy], area]
                            print(f"area={area}")
                    # mouse left button DBL click on face to start tracking
                    cv2.setMouseCallback('Person Tracking', click_on_face, [[x1, y1, x2, y2], id])

        cv2.putText(img, 'No of faces:' + (str(int(f))), (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)
        cv2.namedWindow("Person Tracking")
        return img


if __name__ == "__main__":
    Track = False
    TrackId = None
    fbRange = [6200, 6800]
    pid = [0.4, 0.4, 0]
    previousError = 0
    info = [[0, 0], 0]

    required_shape = (160, 160)
    face_encoder = InceptionResNetV2()
    path_m = "face_recognition/dataset/model/facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = 'face_recognition/dataset/encodings/encodings.pkl'
    face_detector = mtcnn.MTCNN()
    encoding_dict = load_pickle(encodings_path)

    # load PersonTracker class
    person_tracker = PersonTracker()

    # load Deepsort
    face_deepsort = DeepsortFaceTracker()
    deepsort = face_deepsort.tracker()

    # video recorder
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output.avi", fourcc, 15.0, (int(960), int(720)))

    while True:
        # _, frame = cap.read()
        frame = drone.get_frame_read().frame
        h, w, c = frame.shape
        frame = cv2.resize(frame, (w, h))
        # print(w, h) 960 720
        # track person
        frame = person_tracker.person_tracker(frame, face_detector, face_encoder, encoding_dict, face_deepsort,
                                              deepsort)
        if info[1] > 0:
            previousError = trackface(info, w, pid, previousError, fbRange,drone)
            # cv2.putText(frame, "track", (550, 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (0, 200, 200), 2)

        cv2.imshow("Person Tracking", frame)
        #save output video
        video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            drone.land()
            break

    video_writer.release()
    cv2.destroyAllWindows()
