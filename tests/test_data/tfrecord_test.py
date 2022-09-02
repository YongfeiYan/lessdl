import argparse
import os
import time
import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str)
    args = parser.parse_args()
    print('args', args)
    files = glob.glob(args.in_path)
    begin_time = time.time()

    from simdltk.data.tfrecord.reader import example_loader 
    data_sample = None
    for file in files:
        loader = example_loader(file)
        data = list(loader)
        if data_sample is None:
            data_sample = data[0]
        print('length', len(data), 'in', file)
        print('time', (time.time() - begin_time) / 60)

    print(data_sample)
    print('OK!')
