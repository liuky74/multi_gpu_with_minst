from ast import arg
from tqdm import tqdm
import torch
from torch import optim
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.multiprocessing import Process
import torch.distributed as dist
import sys,os

from utils import is_main_process, reduce_value


# 配置文件
class Config():
    # num_classes = 10
    lr = 0.001
    data_dir = "./mnist_data"
    weights = "resnet18.pth"
    device = [0,1]
    world_size = 2
    dist_url= "env://"
    download_data = True
    batch_size = 512
    num_workers=4
    step_size = 10
    max_epoch = 50


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
    

def train_one_epoch(model,optimizer,data_loader,device,epoch):
    model.train()
    # 损失函数
    loss_fun = nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    if is_main_process():
        data_loader = tqdm(data_loader)
    optimizer.zero_grad() # 梯度归零
    for iter_idx,data_tensor in enumerate(data_loader):
        data,target = data_tensor
        # 数据集默认是灰度图,因此只有单通道,你可以选择修改模型的in channel,或者将单通道叠加成3通道,这里是后者
        data= torch.cat([data for _ in range(3)],dim=1)
        data = data.to(device)
        target = target.to(device)
        
        # 正向运算
        logit = model(data)
        loss = loss_fun(logit,target)
        # 反向传播
        loss.backward()
        loss = reduce_value(loss,average=True)
        mean_loss = (mean_loss * iter_idx + loss.detach()) / (iter_idx + 1)  # update mean losses

        # 在进程0中打印平均loss
        if is_main_process():
            message = "|INFO:epoch:{}| {} | loss :{:.4f}|".format(epoch,iter_idx,loss.detach().cpu().numpy())
            data_loader.set_description(message)
        if not torch.isfinite(loss):
            print('|ERR: {} non-finite loss, ending training|'.format(loss))
            sys.exit(1)

        # 优化器步进
        optimizer.step()
        optimizer.zero_grad() # 梯度归零
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    return mean_loss.item()

def evaluate(model,data_loader,device):
    # 测试模式,锁定BN层等参数
    model.eval()
    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)
    if is_main_process():
        data_loader = tqdm(data_loader)
    for iter_idx,(data,target) in enumerate(data_loader):
        data= torch.cat([data for _ in range(3)],dim=1)
        data = data.to(device)
        target = target.to(device)
        logit = model(data)
        pred_cls = torch.argmax(logit,dim=1)
        sum_num += torch.eq(pred_cls,target).count_nonzero()
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    return reduce_value(sum_num,average=False).item()


def run(rank,world_size,args:Config):
    # world_size 表示有多少进程,一条进程表示了一张显卡
    # rank表示当前为第几个进程

    # 初始化各进程

    # 这个必须要设置,
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    args.rank = rank
    args.world_size = world_size
    args.gpu = rank
    torch.cuda.set_device(args.gpu)
    args.dist_backend="nccl"
    print('|INFO: distributed init (rank {}): {}|'.format(args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend,world_size=args.world_size,rank=args.rank)
    dist.barrier()
    # end
    device = torch.device(rank) # 当输入"cuda"时,torch会自动分配device
    batch_size = args.batch_size
    weights_path = args.weights
    args.lr *= args.world_size

    # 从官方例子中下载数据集
    train_data_set = torchvision.datasets.MNIST(root=args.data_dir,download=args.download_data,transform=Transformer(),train=True)
    val_data_set = torchvision.datasets.MNIST(root=args.data_dir,download=args.download_data,transform=Transformer(),train=False)

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    # 构建读取器
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=args.num_workers,
                                               collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=args.num_workers,
                                             collate_fn=collate_fn)

    # 官方预定义模型ResNet18
    model = torchvision.models.resnet.resnet18().to(device)

    # 如果存在预训练权重则载入
    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = "initial_weights.pt"
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()

        # 其他进程模型载入主进程中的初始化模型参数,这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 启用batch data BN同步,会更耗时但是BN效果会更好
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # Adam梯度下降最优化
    optimizer = optim.Adam(
        params=model.parameters(), # 需要更新的参数
        lr= args.lr, # 初始学习率
        betas=(0.9,0.999), # 一阶矩阵更新加权与二阶矩阵更新加权
        )

    # 学习率缩放器,每10个迭代学习率*0.1
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer,
                                            step_size=args.step_size,gamma=0.1)

    for epoch in range(args.max_epoch)              :
        train_sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(model,optimizer,train_loader,device,epoch)

        lr_scheduler.step()

        sum_num = evaluate(model=model,data_loader=val_loader,device=device)
        acc = sum_num / val_sampler.total_size

        if rank == 0:
            print("|INFO : |VAL| epoch:{} |acc: {}|".format(epoch,acc))
    # 删除临时保存的初始化权重
    if rank == 0:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
    # 清除
    dist.destroy_process_group()


if __name__ == '__main__':
    config = Config()
    world_size = config.world_size
    processes=[]
    for rank in range(world_size):
        p = Process(target=run,args=(rank,world_size,config))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


    





    