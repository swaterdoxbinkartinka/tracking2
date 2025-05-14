import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
from . import config

def track_video(video_path: str, output_path: str = 'output_video.mp4', model_path: str = 'yolov8n.pt'):
    model = YOLO(model_path)
    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with tqdm(total=frame_count, desc="Рендер кадров", unit="кадр", ncols=100) as pbar:
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frame += 1
            results = model(frame, verbose=False)[0]
            detections = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                if label in config.TRACK_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                label = track.get_det_class() or 'object'

                cv2.rectangle(frame, (x1, y1), (x2, y2), config.BOX_COLOR, config.BOX_THICKNESS)
                cv2.putText(frame, f"{label} ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, config.TEXT_SCALE, config.TEXT_COLOR, config.TEXT_THICKNESS)

                if config.DRAW_CROSS:
                    cv2.line(frame, (x1, y1), (x2, y2), config.CROSS_COLOR, config.CROSS_THICKNESS)
                    cv2.line(frame, (x2, y1), (x1, y2), config.CROSS_COLOR, config.CROSS_THICKNESS)

            cv2.putText(frame, f"Кадр {current_frame}/{frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out.write(frame)

            if config.SHOW_PREVIEW:
                cv2.imshow(config.PREVIEW_WINDOW_NAME, frame)
                if cv2.waitKey(config.PREVIEW_FRAME_DELAY) & 0xFF == ord('q'):
                    print("\n⛔ Принудительная остановка по 'q'")
                    break

            pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Видео успешно сохранено как '{output_path}'")
