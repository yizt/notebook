

## 视频读写

```python
    import cv2
    cap = cv2.VideoCapture(video_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter()
    writer.open(output_path, fourcc, fps, (w, h), True)
    flag = True
    while flag and cap.isOpened():
        flag, frame = cap.read()
        if frame is None:
            continue
        writer.write(frame)

    cap.release()
    writer.release()
```