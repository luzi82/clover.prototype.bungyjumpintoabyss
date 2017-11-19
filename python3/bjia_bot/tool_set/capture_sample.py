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
    parser.add_argument('--scm_path', nargs='?', help='state classifier model path')
    parser.add_argument('--scm_score', nargs='?', type=float, help='score to output img')
    parser.add_argument('--scm_img_path', nargs='?', help='scm img path')
    args = parser.parse_args()
    
    clover.tool_set.capture_sample.capture_sample(FFMPEG_EXEC_PATH, args.src_name, WIDTH, HEIGHT, OUTPUT_FOLDER,
        args.disable_sleep, args.scm_path, args.scm_score, args.scm_img_path)
