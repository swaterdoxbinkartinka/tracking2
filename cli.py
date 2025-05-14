import argparse
from yolo_tracker.tracker import track_video

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 + DeepSORT видео-трекинг")
    parser.add_argument("video_path", help="Путь к входному видео")
    parser.add_argument("--output", default="output_video.mp4", help="Путь к выходному видео")
    parser.add_argument("--model", default="yolov8n.pt", help="Путь к модели YOLOv8")
    args = parser.parse_args()
    track_video(args.video_path, args.output, args.model)
