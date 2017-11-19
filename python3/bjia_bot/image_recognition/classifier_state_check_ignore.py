import os
import clover.common
import clover.image_recognition
import bjia_bot.image_recognition.classifier_state as classifier_state

def should_not_ignore(filename):
    img = classifier_state.load_img(filename)
    return not classifier_state.should_ignore(img)

if __name__ == '__main__':
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description='state')
    parser.add_argument('state', nargs='?', help='state')
    args = parser.parse_args()

    if args.state != None:
        state_list = [args.state]
    else:
        state_list = clover.image_recognition.get_label_state_list()

    clover.common.reset_dir('output')
    for state in state_list:
        filename = os.path.join('image_recognition','label','state','{}.txt'.format(state))
        file_list = clover.common.readlines(filename)
        file_list = list(filter(should_not_ignore,file_list))
        clover.common.writelines(filename, file_list)
