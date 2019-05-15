import os
import torch
import time
import torchvision as tv
from model import NetD, NetG
from torch.autograd import Variable


class Config:
    data_path = 'D:\\data\\' if os.name == 'nt' else './data/'
    num_workers = 4 # 多进程加载数据所用的进程数
    image_size = 96
    batch_size = 256
    max_epoch = 5
    lr_netg = 2e-4
    lr_netd = 2e-4
    ngf = 64    # 生成器 feature map 数
    ndf = 64    # 判别器 feature map 数
    nz = 100    # 噪声维度
    beta1 = 0.5 # Adam优化器的参数

    g_every = 5 # 每 5个batch训练一次生成器
    d_every = 1 # 每 1个batch训练一次判别器
    save_every = 10 # 每 10个batch保存一次
    netd_path = "D:\\qqfile\\netd.pkl"
    netg_path = "D:\\qqfile\\netg.pkl"
    gpu = True

    vis = False
    env = "GAN"
    plot_every = 20

    # 测试
    gen_num = 128
    gen_best = 8
    gen_mean = 0  # 噪声均值
    gen_std = 1   # 噪声方差
    gen_img = 'generated.png'


def train(**kwargs):
    print("Start training...")
    tic = time.time()
    for k, v in kwargs.items():
        setattr(opt, k, v)

    for epoch in range(opt.max_epoch):
        print('epoch %d start...' % epoch, end='')
        toc = time.time()
        for i, (img, _) in enumerate(dataloader):
            # 加载数据
            real_img = Variable(img)
            if opt.gpu:
                real_img = real_img.cuda()

            # 训练判别器
            if (i + 1) % opt.d_every == 0:
                optimizer_d.zero_grad()

                ## 尽可能标记真照片为 1
                output = netd(real_img)
                err_d_real = criterion(output, true_labels)
                err_d_real.backward()

                ## 尽可能标记假照片为 0
                noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises).detach()  # 防止 noise 梯度反传回 netg
                output = netd(fake_img)
                err_d_fake = criterion(output, fake_labels)
                err_d_fake.backward()
                optimizer_d.step()
                # 可视化
                #err_d = err_d_fake + err_d_real
                #err_d_meter.add(err_d.data[0])

            # 训练生成器
            if (i + 1) % opt.g_every == 0:
                optimizer_g.zero_grad()
                ## 尽可能骗过判别器
                noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                err_g = criterion(output, true_labels)
                err_g.backward()
                optimizer_g.step()
                # 可视化
                #err_g_meter.add(err_g.data[0])
        print('finished. Cost %.2f s' % (time.time()-toc))

    print("Finish training. Cost %.2f min" % ((time.time()-tic)/60))

    loc = time.localtime()
    torch.save(netd.state_dict(), "netd_%d_%d.pkl" % (loc.tm_mon, loc.tm_mday))
    torch.save(netg.state_dict(), "netg_%d_%d.pkl" % (loc.tm_mon, loc.tm_mday))


def generate(**kwargs):
    print("Start drawing...")
    for k, v in kwargs.items():
        setattr(opt, k, v)

    noises = torch.randn(opt.gen_num, opt.nz, 1, 1)
    noises.normal_(opt.gen_mean, opt.gen_std)  # 正则化
    noises = Variable(noises)
    if opt.gpu:
        noises = noises.cuda()

    fake_img = netg(noises)
    scores = netd(fake_img).data

    indexs = scores.topk(opt.gen_best)[1]
    result = []
    for idx in indexs:
        result.append(fake_img.data[idx])


    # 保存图片
    ## stack : 最后一个维度的元素相叠加，形成一个新的tensor
    tv.utils.save_image(torch.stack(result), filename=opt.gen_img, normalize=True, range=(-1, 1))
    print("Finished!")


if __name__ == "__main__":
    opt = Config()

    if opt.vis:
        from visualize import Visualizer
        vis = Visualizer(opt.env)

    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        drop_last=True
    )

    netg = NetG(opt)
    netd = NetD(opt)

    # 加载已有的网络参数
    if opt.netd_path:
        print('Loading netd...', end='')
        netd.load_state_dict(torch.load(opt.netd_path))
        print('Successful!')
    if opt.netg_path:
        print('Loading netg...', end='')
        netg.load_state_dict(torch.load(opt.netg_path))
        print('Successful!')

    optimizer_g = torch.optim.Adam(netg.parameters(), opt.lr_netg, betas=(opt.beta1, 0.999))
    optimizer_d = torch.optim.Adam(netd.parameters(), opt.lr_netd, betas=(opt.beta1, 0.999))

    criterion = torch.nn.BCELoss()

    # 真图片 label 为 1，假图片为 0
    true_labels = Variable(torch.ones(opt.batch_size))
    fake_labels = Variable(torch.zeros(opt.batch_size))

    #fix_noises = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1))
    noises = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1))

    if opt.gpu:
        netg.cuda()
        netd.cuda()
        criterion.cuda()
        true_labels = true_labels.cuda()
        fake_labels = fake_labels.cuda()
        #fix_noises = fix_noises.cuda()
        noises = noises.cuda()

    while True:
        action = input('Train(t) or Generate(g) or Quit(q)> ').lower()

        if action == 't':
            epochs = int(input('Epoch times > '))
            train(max_epoch=epochs)

        elif action == 'g':
            num_imgs = int(input('How many images > '))
            generate(gen_best=num_imgs)

        elif action == 'q':
            print('Bye~')
            break
