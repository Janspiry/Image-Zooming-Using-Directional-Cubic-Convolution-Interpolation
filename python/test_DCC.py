import numpy as np
from PIL import Image
import argparse
import multiprocessing
from functools import partial
from tqdm import tqdm
import os
from pathlib import Path
from DCC import DCC
# uniform format image
def numpy2img(img, out_type=np.uint8, min_max=(0, 1)):
    img = img.clip(*min_max)  # clamp
    img = (img - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    if out_type == np.uint8:
        img_np = (img * 255.0).round()
    return img_np.astype(out_type)

# resize image by DCC algorithm
def resize_worker(img_file, level):
    # Image to numpy
    img = Image.open(img_file).convert('RGB')
    img = np.array(img).astype(np.float)
    img = img/255
    sr_img = DCC(img, level)
    # return image names and processsd image
    return img_file.name.split('.')[0], Image.fromarray(numpy2img(sr_img))

def prepare(img_path, out_path, n_worker, level):
    resize_fn = partial(resize_worker, level=level)

    files = [p for p in Path(f'{img_path}').glob(f'**/*')]
    os.makedirs(out_path, exist_ok=True)
    total = 0
    for img in tqdm(files):
        i, img = resize_fn(img)
        img.save(
            '{}/{}_{}.png'.format(out_path, i.zfill(5), 'dcc_numpy'))
        total += 1

# multiprocessing support
def prepare_mp(img_path, out_path, n_worker, level):
    resize_fn = partial(resize_worker, level=level)

    files = [p for p in Path(f'{img_path}').glob(f'**/*')]
    os.makedirs(out_path, exist_ok=True)

    total = 0
    with multiprocessing.Pool(n_worker) as pool:
        for i, img in tqdm(pool.imap_unordered(resize_fn, files)):
            img.save(
                '{}/{}_{}.png'.format(out_path, i.zfill(5), 'dcc_numpy'))
            total += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        default='./data/hr')
    parser.add_argument('--out', '-o', type=str, default='./data/sr')

    parser.add_argument('--level', type=int, default=2)
    parser.add_argument('--n_worker', type=int, default=8)

    args = parser.parse_args()
    # args.out = '{}_{}_{}'.format(args.out, 'dcc_numpy', 2**args.level)

    prepare(args.path, args.out, args.n_worker, level=args.level)
