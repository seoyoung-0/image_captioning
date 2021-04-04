from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    # 함수 실행
    create_input_files(dataset='coco',
                       karpathy_json_path='../../1. data/MSCOCO_Korean.json',
                       image_folder='../../1. data',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='../../1. data/temp',
                       max_len=50)
