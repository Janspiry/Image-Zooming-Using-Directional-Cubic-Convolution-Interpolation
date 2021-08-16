## Image Zooming Using Directional Cubic Convolution Interpolation

[Paper](https://digital-library.theiet.org/content/journals/10.1049/iet-ipr.2011.0534) 



### Brief

This is a unoffical python implementation about **Image Zooming Using Directional Cubic Convolution Interpolation (DCC)** by **Numpy**, which claims it can preserve the sharp edges and details of
images with noticeable suppression of the artifacts that usually occur with cubic convolution interpolation.  



### Status

Compared to the open source MATLAB version, we are ready to do the following :

-   [x] Python Version
-   [x] RGB Image support 
-   [x] Multi Process support
-   [x] Eliminate edge white points



### Result

| Bilinear                                        | Bicubic                                       | DCC                                               |
| ----------------------------------------------- | --------------------------------------------- | ------------------------------------------------- |
| ![00000_bilinear](./data/sr/00000_bilinear.png) | ![00000_bicubic](./data/sr/00000_bicubic.png) | ![00000_dcc_numpy](./data/sr/00000_dcc_numpy.png) |
| ![00001_bilinear](./data/sr/00001_bilinear.png) | ![00001_bicubic](./data/sr/00001_bicubic.png) | ![00001_dcc_numpy](./data/sr/00001_dcc_numpy.png) |
| ![00002_bilinear](./data/sr/00002_bilinear.png) | ![00002_bicubic](./data/sr/00002_bicubic.png) | ![00002_dcc_numpy](./data/sr/00002_dcc_numpy.png) |



###  Usage

We have integrated the open source version of MATLAB and the manually implemented version of Python, which can be found in corresponding folder. Take the python version for a example:

```python
from DCC import DCC
img = Image.open(img_file).convert('RGB')
img = np.array(img).astype(np.float)/255
sr_img = DCC(img, level)
```

*Note*: DCC get the low resolution image first by interval sampling in MATLAB version, which is not the same with general method. You can change following code to use different down-sample methods or just use the low resolution image as input.

```python
def DCC(img, level):
    # get the low resolution image by interval sampling
    lr_img = img[0:-1:2**level, 0:-1:2**level, :]
    sr_img = img
```



### Acknowledges

1. [IET Digital Library: Image zooming using directional cubic convolution interpolation (theiet.org)](https://digital-library.theiet.org/content/journals/10.1049/iet-ipr.2011.0534)
2. https://www.mathworks.com/matlabcentral/fileexchange/38570-image-zooming-using-directional-cubic-convolution-interpolation



