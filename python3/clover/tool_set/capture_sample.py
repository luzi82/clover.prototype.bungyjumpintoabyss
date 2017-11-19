from clover.video_input import video_capture
import random
import os
import time
import sys
import cv2
import clover.common
import shutil

def capture_sample(arg_ffmpeg_exec_path, arg_src_name, arg_width, arg_height, arg_output_folder,
    arg_disable_sleep=False, arg_classifier=None, arg_label_output_path=None):

    t = int(time.time()*1000)
    output_folder = os.path.join(arg_output_folder,str(t))
    clover.common.makedirs(output_folder)

    vc = video_capture.VideoCapture(arg_ffmpeg_exec_path,arg_src_name,arg_width,arg_height)
    vc.start()
    vc.wait_data_ready()
    while True:
        t = int(time.time()*1000)
        t0 = int(t/100000)
        ndata = vc.get_frame()
        write_ok = True
        if arg_classifier:
            label, perfect = arg_classifier.get_state(ndata.astype('float32')*2/255-1)
            print('label={}, perfect={}'.format(label,perfect),file=sys.stderr)
            write_ok = write_ok and (not perfect)
        if write_ok:
            fn_dir = os.path.join(output_folder,str(t0))
            fn = os.path.join(fn_dir,'{}.png'.format(t))
            clover.common.makedirs(fn_dir)
            print(fn,file=sys.stderr)
            cv2.imwrite(fn,ndata)
        if write_ok and (arg_classifier):
            ffn_dir = os.path.join(arg_label_output_path, label)
            ffn = os.path.join(ffn_dir,'{}.png'.format(t))
            clover.common.makedirs(ffn_dir)
            print(ffn,file=sys.stderr)
            shutil.copyfile(fn,ffn)
        vc.release_frame()
        if not arg_disable_sleep:
            time.sleep(0.05+0.05*random.random())
    vc.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='video capture')
    parser.add_argument('ffmpeg_exec_path', help='ffmpeg_exec_path')
    parser.add_argument('src_name', help='src_name')
    parser.add_argument('width', type=int, help='width')
    parser.add_argument('height', type=int, help='height')
    parser.add_argument('output_folder', help='output_folder')
    parser.add_argument('--disable_sleep', action='store_true', help='disable_sleep')
    args = parser.parse_args()
    
    capture_sample(args.ffmpeg_exec_path, args.src_name, args.width, args.height, args.output_folder,
        args.disable_sleep)
