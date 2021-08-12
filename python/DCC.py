import numpy as np
from PIL import Image
import argparse
import multiprocessing
from functools import partial
from tqdm import tqdm
import os
from pathlib import Path
def DetectDirect(A, type, k, T):
    if type == 1:
        # 45 degree diagonal direction
        t1 = abs(A[2,0]-A[0,2])   
        t2 = abs(A[4,0]-A[2,2])+abs(A[2,2]-A[0,4])     
        t3 = abs(A[6,0]-A[4,2])+abs(A[4,2]-A[2,4])+abs(A[2,4]-A[0,6]) 
        t4 = abs(A[6,2]-A[4,4])+abs(A[4,4]-A[2,6]) 
        t5 = abs(A[6,4]-A[4,6]) 
        d1 = t1+t2+t3+t4+t5
        
        # 135 degree diagonal direction
        t1 = abs(A[0,4]-A[2,6]);   
        t2 = abs(A[0,2]-A[2,4])+abs(A[2,4]-A[4,6]);   
        t3 = abs(A[0,0]-A[2,2])+abs(A[2,2]-A[4,4])+abs(A[4,4]-A[6,6]); 
        t4 = abs(A[2,0]-A[4,2])+abs(A[4,2]-A[6,4])
        t5 = abs(A[4,0]-A[6,2])
        d2 = t1+t2+t3+t4+t5
    else:
        # horizontal direction
        t1 = abs(A[0,1]-A[0,3])+abs(A[2,1]-A[2,3])+abs(A[4,1]-A[4,3])
        t2 = abs(A[1,0]-A[1,2])+abs(A[1,2]-A[1,4])
        t3 = abs(A[3,0]-A[3,2])+abs(A[3,2]-A[3,4])
        d1 = t1+t2+t3   
        
        # vertical direction
        t1 = abs(A[1,0]-A[3,0])+abs(A[1,2]-A[3,2])+abs(A[1,4]-A[3,4])
        t2 = abs(A[0,1]-A[2,1])+abs(A[2,1]-A[4,1])
        t3 = abs(A[0,3]-A[2,3])+abs(A[2,3]-A[4,3])
        d2 = t1+t2+t3
    # Compute the weight vector
    w = np.array([1/(1+d1**k), 1/(1+d2**k)])
    # Compute the directional index
    n = 3
    if (1+d1)/(1+d2) > T:
        n = 1
    elif (1+d2)/(1+d1) > T:
        n = 2
    return w, n

def PixelValue(A, type, w, n, f):
    if type == 1:
        v1 = np.diag(np.fliplr(A))[::2]
        v2 = np.diag(A)[::2]
    else:
        v1 =  A[3,::2]
        v2 =  A[::2,3]
    if n == 1:
        p = np.dot(v2, f)
    elif n == 2:
        p = np.dot(v1, f)
    else:
        p1 = np.dot(v1, f)
        p2 = np.dot(v2, f)
        p = (w[0]*p1+w[1]*p2)/(w[0]+w[1])
    return p
def _DCC(I, k, T):
    m, n = I.shape
    nRow = 2*m
    nCol = 2*n
    A = np.zeros([nRow, nCol])
    A[0:-1:2, 0:-1:2] = I
    f = np.array([-1, 9, 9, -1])/16
    for i in range(3,nRow-4,2):
        for j in range(3,nCol-4,2):
            [w,n] = DetectDirect(A[i-3:i+4,j-3:j+4],1,k,T)
            A[i,j] = PixelValue(A[i-3:i+4,j-3:j+4],1,w,n,f)
    for i in range(4,nRow-5,2):
        for j in range(3,nCol-4,2):
            [w,n] = DetectDirect(A[i-2:i+3,j-2:j+3],2,k,T)
            A[i,j] = PixelValue(A[i-3:i+4,j-3:j+4],2,w,n,f)
    for i in range(3,nRow-4,2):
        for j in range(4,nCol-5,2):
            [w,n] = DetectDirect(A[i-2:i+3,j-2:j+3],3,k,T)
            A[i,j] = PixelValue(A[i-3:i+4,j-3:j+4],3,w,n,f)
    return A                

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
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # hyper parameters
    lr_img = img[0:-1:2**level, 0:-1:2**level, :]
    k, T = 5, 1.15
    sr_img = img
    for channel in range(lr_img.shape[-1]):
        sr_img_simple = lr_img[:,:,channel]
        for _ in range(level):
            sr_img_simple  = _DCC(sr_img_simple , k, T)
        sr_img[:,:,channel] = sr_img_simple
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
