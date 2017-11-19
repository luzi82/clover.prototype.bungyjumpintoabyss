from clover.video_input import video_capture
import random
import os
import time
import sys
import cv2

def capture_sample(arg_ffmpeg_exec_path, arg_src_name, arg_width, arg_height, arg_output_folder,
    arg_disable_sleep=False, arg_scm_path=None, arg_scm_score=None, arg_scm_img_path=None):

    assert( (not arg_disable_sleep) or ( arg_scm_score != None ) )
    assert( (arg_scm_score != None) == (arg_scm_path != None) )
    assert( (arg_scm_img_path != None) == (arg_scm_path != None) )

    t = int(time.time()*1000)
    output_folder = os.path.join(arg_output_folder,str(t))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    state_clr = None
    if arg_scm_path:
        import classifier_state
        state_clr = classifier_state.StateClassifier(arg_scm_path)
        scm_img_path = os.path.join(arg_scm_img_path,str(t))

    vc = video_capture.VideoCapture(arg_ffmpeg_exec_path,arg_src_name,arg_width,arg_height)
    vc.start()
    vc.wait_data_ready()
    while True:
        t = int(time.time()*1000)
        t0 = int(t/100000)
        ndata = vc.get_frame()
        write_ok = True
        if state_clr:
            label, score = state_clr.get_state(ndata.astype('float32')*2/255-1)
            print('{} {}'.format(label,score),file=sys.stderr)
            write_ok = write_ok and (score < arg_scm_score)
            if write_ok:
                ffn_dir = os.path.join(scm_img_path, label)
                ffn = os.path.join(ffn_dir,'{}.png'.format(t))
                if not os.path.isdir(ffn_dir):
                    os.makedirs(ffn_dir)
                print(ffn,file=sys.stderr)
                cv2.imwrite(ffn,ndata)
        if write_ok:
            fn_dir = os.path.join(output_folder,str(t0))
            fn = os.path.join(fn_dir,'{}.png'.format(t))
            if not os.path.isdir(fn_dir):
                os.makedirs(fn_dir)
            print(fn,file=sys.stderr)
            cv2.imwrite(fn,ndata)
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
    parser.add_argument('--scm_path', nargs='?', help='state classifier model path')
    parser.add_argument('--scm_score', nargs='?', type=float, help='score to output img')
    parser.add_argument('--scm_img_path', nargs='?', help='scm img path')
    args = parser.parse_args()
    
    capture_sample(args.ffmpeg_exec_path, args.src_name, args.width, args.height, args.output_folder,
        args.disable_sleep, args.scm_path, args.scm_score, args.scm_img_path)
