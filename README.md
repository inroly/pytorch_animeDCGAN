# pytorch_animeDCGAN
这是一个使用pytorch完成DCGAN生成动漫人物图像的机器学习代码

![generated](https://github.com/inroly/pytorch_animeDCGAN/blob/master/page_png/generated.png "generated")

这个是训练器迭代了一轮后生成出来的图片，还是有人形的。

---
##首先是pytorch的安装环节

需要一个良好的网络环境
[pytorch](https://pytorch.org/)按照官网给的连接选择自己适合的型号然后安装

![pytorch](https://github.com/inroly/pytorch_animeDCGAN/blob/master/page_png/pytorch_choose.png "pytorch")

这个时候我们需要了解我们电脑的cuda型号

![nvidia1](https://github.com/inroly/pytorch_animeDCGAN/blob/master/page_png/nvidia1.png "nvidia1")

先进入控制面板进入nvidia

![nvidia2](https://github.com/inroly/pytorch_animeDCGAN/blob/master/page_png/nvidia2.png "nvidia2")

查看系统信息中的组件里面有个cuda的版本号，然后对照这下载就可以了

然后就可以使用pytorch中给出的安装指令来执行，但是限制于网络状况，在安装过程中多此出现socket：read time out的问题。
所以直接将whl下载到本地然后使用pip install PATH来安装pytorch，成功

然后有很多依赖包都是要先下载pytorch再下载使用的，所需要的包的为visdom，fire，torchvision，tqdm，ipdb。由于未出现错误，所以不多说。

---
再是图片，
[图片](https://pan.baidu.com/s/1eSifHcA)
 提取码：g5qa 
感谢知乎用户何之源爬取的数据。
```
由于本次代码是文件名作为类名，所以格式为
//data//face//.png
路径可以自己在代码里面修改
```
