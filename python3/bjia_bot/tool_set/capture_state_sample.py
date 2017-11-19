import clover.tool_set.capture_sample
import os
import bjia_bot.image_recognition.classifier_state as clr_state

FFMPEG_EXEC_PATH = '/usr/bin/ffmpeg'
WIDTH = 320
HEIGHT = 180
OUTPUT_FOLDER = os.path.join('image_recognition','raw_image')
STATE_OUTPUT_FOLDER = os.path.join('output')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='video capture')
    parser.add_argument('src_name', help='src_name')
    parser.add_argument('--disable_sleep', action='store_true', help='disable_sleep')
    args = parser.parse_args()
    
    state_classifier = clr_state.StateClassifier(clr_state.MODEL_PATH)
    
    clover.tool_set.capture_sample.capture_sample(FFMPEG_EXEC_PATH, args.src_name, WIDTH, HEIGHT, OUTPUT_FOLDER,
        args.disable_sleep, state_classifier, STATE_OUTPUT_FOLDER)
