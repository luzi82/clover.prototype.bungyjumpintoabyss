import clover.tool_set.capture_sample
import os

FFMPEG_EXEC_PATH = '/usr/bin/ffmpeg'
WIDTH = 320
HEIGHT = 180
OUTPUT_FOLDER = os.path.join('image_recognition','raw_image')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='video capture')
    parser.add_argument('src_name', help='src_name')
    parser.add_argument('--disable_sleep', action='store_true', help='disable_sleep')
    args = parser.parse_args()
    
    clover.tool_set.capture_sample.capture_sample(FFMPEG_EXEC_PATH, args.src_name, WIDTH, HEIGHT, OUTPUT_FOLDER,
        args.disable_sleep)
