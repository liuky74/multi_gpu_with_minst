from tqdm import tqdm
import torch
from torch import optim
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader



# 配置文件
class Config():
    # num_classes = 10
    lr = 0.001
    data_dir = "./mnist_data"
    download_data = True
    batch_size = 32
    num_workers=4
    step_size = 10
    max_epoch = 50

config = Config()

# 图像增强
class Transformer():
    def __init__(self) -> None:
        self.transformer = transforms.Compose([
            # 随机水平翻转
            transforms.RandomHorizontalFlip(),
            # 随机垂直翻转
            transforms.RandomVerticalFlip(),
            # 随机角度旋转
            transforms.RandomRotation(degrees=15),
            # 缩放
            transforms.Resize(112),
            # 转换为Tensor
            transforms.ToTensor()
        ])
        
    def __call__(self,input):
        return self.transformer(input)
# 后处理
def collate_fn(batch_data):
    imgs=[]
    targets=[]
    for img,target in batch_data:
        imgs.append(img[None,...])
        targets.append(target)
    imgs = torch.cat(imgs,dim=0)
    targets = torch.Tensor(targets)
    return imgs,targets.long()
    

if __name__ == '__main__':
    # 从官方例子中下载数据集
    mnist_dataset_train = torchvision.datasets.MNIST(root=config.data_dir,download=config.download_data,transform=Transformer(),train=True)
    mnist_dataset_test = torchvision.datasets.MNIST(root=config.data_dir,download=config.download_data,transform=Transformer(),train=False)
    # 构建读取器
    data_loader_train = DataLoader(
                            mnist_dataset_train,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            collate_fn=collate_fn
                            )
    data_loader_test = DataLoader(
                            mnist_dataset_test,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            collate_fn=collate_fn
                            )

    # 官方预定义模型ResNet18
    model = torchvision.models.resnet.resnet18(True)
    model = model.cuda() # 从内存存入显存

    # Adam梯度下降最优化
    optimizer = optim.Adam(
        params=model.parameters(), # 需要更新的参数
        lr= config.lr, # 初始学习率
        betas=(0.9,0.999), # 一阶矩阵更新加权与二阶矩阵更新加权
        )
    # 学习率缩放器,每10个迭代学习率*0.1
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer,
                                            step_size=config.step_size,gamma=0.1)
    # 损失函数
    loss_fun = nn.CrossEntropyLoss()
    for epoch_idx in range(config.max_epoch):
        # 训练模式,BN层更新
        model.train()
        process_bar = tqdm(enumerate(data_loader_train))
        for iter_idx,(data,target) in enumerate(data_loader_train):
            # 数据集默认是灰度图,因此只有单通道,你可以选择修改模型的in channel,或者将单通道叠加成3通道,这里是后者
            data= torch.cat([data for _ in range(3)],dim=1)
            data = data.cuda() # 从内存存入显存
            target = target.cuda() # 从内存存入显存
            # 梯度归零
            optimizer.zero_grad()
            # 正向运算
            logit = model(data)
            loss = loss_fun(logit,target)
            # 反向传播
            loss.backward()
            # 优化器步进
            optimizer.step()
            if iter_idx % 20 ==19:
                message = "|INFO:epoch:{}| {}/{} | loss :{:.4f}|".format(epoch_idx,iter_idx,(len(mnist_dataset_train)//config.batch_size),loss.detach().cpu().numpy())
                process_bar.set_description(message)
            pass
        # 学习率缩放器步进
        lr_scheduler.step()
        
        # 测试模式,锁定BN层等参数
        model.eval()
        acc = 0
        process_bar = tqdm(enumerate(data_loader_test))
        for iter_idx,(data,target) in enumerate(data_loader_test):
            data= torch.cat([data for _ in range(3)],dim=1)
            data = data.cuda()
            target = target.cuda()
            logit = model(data)
            pred_cls = torch.argmax(logit,dim=1)
            acc_ = torch.eq(pred_cls,target).count_nonzero()/target.shape[0]
            acc+=acc_
            if iter_idx % 20 ==19:
                message = "|INFO: acc: {}|".format(acc/(iter_idx+1))
                process_bar.set_description(message)
            

        # 学习率缩放器步进

    