import ffmpeg
import numpy as np


def read_frames(video_path, NUM_FRAMES, start=None, end=None, fps=10, safe_read=False):
    try:
        probe = ffmpeg.probe(video_path)
        width = int(probe['streams'][0]['width'])
        height = int(probe['streams'][0]['height'])
        if start is None:
            start = 0
            end = float(probe['format']['duration'])

        num_sec = end - start

        start = start + (num_sec / NUM_FRAMES) / 2
        num_sec = num_sec - (num_sec / NUM_FRAMES)

        cmd = (
            ffmpeg
                .input(video_path, ss=start, t=num_sec + 0.1)
                .filter('fps', fps=fps)
        )

        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

        if NUM_FRAMES == 1:
            step = 1
        else:
            step = (len(video) - 1) / (NUM_FRAMES - 1)

        frame_ids = np.arange(0, len(video), step).astype(int)
        video_frames = video[frame_ids]
        video_frames = video_frames[:NUM_FRAMES]
    except Exception as e:
        if isinstance(e, ffmpeg.Error):
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
        else:
            print(e)
        if not safe_read:
            raise e
        else:
            video_frames = np.zeros((8, 224, 224, 3), dtype=np.uint8)

    if len(video_frames) != NUM_FRAMES:
        n, h, w, c = video_frames.shape
        video_frames = np.concatenate((video_frames, np.zeros((NUM_FRAMES - n, h, w, c), dtype=np.uint8)), axis=0)
    return video_frames

