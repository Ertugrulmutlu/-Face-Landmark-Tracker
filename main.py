import cv2 as cv
from Utils.helpers import Helpers
from Utils.detector import Detector

# -------------------- USER SETTINGS --------------------
INPUT_VIDEO  = r"./video/example.mp4"
OUTPUT_VIDEO = r"./output/annotated.mp4"
SAVE_CSV     = True
CSV_PATH     = r"./output/landmarks.csv"
CONF_THRESH  = 0.6
MAX_FACES    = 1
SKIP         = 0
PROTO_PATH   = r"./model/deploy.prototxt"
WEIGHTS_PATH = r"./model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
LBF_MODEL    = r"./model/lbfmodel.yaml"
# -------------------------------------------------------

def main():
    Helpers.ensure_dirs(OUTPUT_VIDEO, CSV_PATH, SAVE_CSV)
    Helpers.check_files([INPUT_VIDEO, PROTO_PATH, WEIGHTS_PATH, LBF_MODEL])

    cap, writer, fps, w, h = Helpers.open_video(INPUT_VIDEO, OUTPUT_VIDEO)

    # Load detector + facemark
    det = Detector(PROTO_PATH, WEIGHTS_PATH)
    det.load_face_detector()
    det.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    det.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    facemark = cv.face.createFacemarkLBF()
    facemark.loadModel(LBF_MODEL)

    csv_file, csv_writer = (None, None)
    if SAVE_CSV:
        csv_file, csv_writer = Helpers.init_csv(CSV_PATH)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        skip_flag = (SKIP > 0 and (frame_idx % (SKIP + 1)) != 0)

        frame = Helpers.process_frame(frame, frame_idx, det, facemark,
                                    CONF_THRESH, MAX_FACES,
                                    csv_writer, SAVE_CSV,
                                    pad_scale=1.25,
                                    draw_boxes=True,
                                    connect=True,
                                    skip=skip_flag,     
                                    alpha=0.7)     

        writer.write(frame)
        frame_idx += 1

    cap.release(); writer.release()
    if csv_file: csv_file.close()
    print(f"Done. Saved: {OUTPUT_VIDEO}" + (f" | CSV: {CSV_PATH}" if SAVE_CSV else ""))

if __name__ == "__main__":
    main()
