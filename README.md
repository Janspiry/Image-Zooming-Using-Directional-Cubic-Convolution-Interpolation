## Image Zooming Using Directional Cubic Convolution Interpolation

[Paper](https://digital-library.theiet.org/content/journals/10.1049/iet-ipr.2011.0534) 



### Brief

This is a unoffical python implementation about **Image Zooming Using Directional Cubic Convolution Interpolation (DCC)** by **Numpy**, which claims it can preserve the sharp edges and details of
images with noticeable suppression of the artifacts that usually occur with cubic convolution interpolation.  



### Status

Compared to the open source MATLAB version, we are ready to do the following :

-   [x] Python Version
-   [x] RGB Image support 
-   [x] MultiProcessing support
-   [ ] Eliminate edge white points



### Result

| Samples\|Methods | Bilinear                                        | Bicubic                                       | DCC                                               |
| ---------------- | ----------------------------------------------- | --------------------------------------------- | ------------------------------------------------- |
| Sample I         | ![00000_bilinear](./data/sr/00000_bilinear.png) | ![00000_bicubic](./data/sr/00000_bicubic.png) | ![00000_dcc_numpy](./data/sr/00000_dcc_numpy.png) |
| Sample II        | ![00001_bilinear](./data/sr/00001_bilinear.png) | ![00001_bicubic](./data/sr/00001_bicubic.png) | ![00001_dcc_numpy](./data/sr/00001_dcc_numpy.png) |
| Sample III       | ![00002_bilinear](./data/sr/00002_bilinear.png) | ![00002_bicubic](./data/sr/00002_bicubic.png) | ![00002_dcc_numpy](./data/sr/00002_dcc_numpy.png) |



###  Usage

We have integrated the open source version of MATLAB and the manually implemented version of Python, which can be found in corresponding folder.



### Acknowledges

1. [IET Digital Library: Image zooming using directional cubic convolution interpolation (theiet.org)](https://digital-library.theiet.org/content/journals/10.1049/iet-ipr.2011.0534)
2. https://www.mathworks.com/matlabcentral/fileexchange/38570-image-zooming-using-directional-cubic-convolution-interpolation



