import csv
import os
import torch.optim as optim
import itertools
from tensorboardX import SummaryWriter
from datetime import datetime
from torchvision.transforms import transforms
from tqdm import tqdm
import time
import argparse
from yolo2 import load_data
from utils import *
from cfg import get_cfgs
from cfg import get_cfg
from tps_grid_gen import TPSGridGen
from generator_dim import GAN_dis
from  loadmodel_v3 import load_yolo3
from constants import Constants

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--net', default='yolov3', help='target net name')
parser.add_argument('--method', default='RFA-Tex', help='method name')
parser.add_argument('--suffix', default=None, help='suffix name')
parser.add_argument('--gen_suffix', default=None, help='generator suffix name')
parser.add_argument('--epoch', type=int, default=None, help='')
parser.add_argument('--z_epoch', type=int, default=None, help='')
parser.add_argument('--device', default='cuda:0', help='')

pargs = parser.parse_args()
args, kwargs = get_cfgs(pargs.net, pargs.method)
RFATexkwargs = get_cfg(pargs.method)
if pargs.epoch is not None:
    args.n_epochs = pargs.epoch
if pargs.z_epoch is not None:
    args.z_epochs = pargs.z_epoch
if pargs.suffix is None:
    pargs.suffix = pargs.net + '_' + pargs.method
device = torch.device(pargs.device)
#yolo3定义
model_path = kwargs['model_path']
weights_path = kwargs['weights_path']
darknet_model_3 = load_yolo3(model_path, weights_path)
darknet_model_3 = darknet_model_3.eval()
darknet_model_3.to(device)
class_names = utils.load_class_names(kwargs['classes_path'])
img_dir_train = kwargs['img_dir_train']
lab_dir_train_v2 = kwargs['lab_dir_train_v2']
lab_dir_train_v3 = kwargs['lab_dir_train_v3']
#v3需要的数据
train_data_v3 = load_data.InriaDataset(img_dir_train, lab_dir_train_v3, kwargs['max_lab'], args.img_size, shuffle=True)
train_loader_v3 = torch.utils.data.DataLoader(train_data_v3, batch_size=kwargs['batch_size'], shuffle=True, num_workers=0)
target_func = lambda obj, cls: obj
patch_applier = load_data.PatchApplier().to(device)
patch_transformer = load_data.PatchTransformer().to(device)
if kwargs['name'] == 'ensemble':
    prob_extractor_yl2 = load_data.MaxProbExtractor(0, 80, target_func, 'yolov2').to(device)
    prob_extractor_yl3 = load_data.MaxProbExtractor(0, 80, target_func, 'yolov3').to(device)
else:
    prob_extractor = load_data.MaxProbExtractor(0, 80, target_func, kwargs['name']).to(device)
total_variation = load_data.TotalVariation().to(device)
#引入Deterioration_Function类
Deterioration_Function = load_data.Deterioration_Function().to(device)
target_control_points = torch.tensor(list(itertools.product(
    torch.arange(-1.0, 1.00001, 2.0 / 4),
    torch.arange(-1.0, 1.00001, 2.0 / 4),
)))
tps = TPSGridGen(torch.Size([300, 300]), target_control_points)
tps.to(device)
target_func = lambda obj, cls: obj
prob_extractor = load_data.MaxProbExtractor(0, 80, target_func, kwargs['name']).to(device)
results_dir = kwargs['result_dir'] + pargs.suffix
print(results_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
#v3loader
loader_v3 = train_loader_v3
epoch_length = len(loader_v3)

def train_RFA_GEN():
    #加载并继续训练的
    gen = GAN_dis(DIM=args.DIM, z_dim=args.z_dim, img_shape=args.patch_size)
    suffix_load = RFATexkwargs['suffix_load']
    result_dir = RFATexkwargs['result_dir']
    rfa_path = os.path.join(result_dir, suffix_load + '.pkl')
    if os.path.exists(rfa_path):
        print(f"Loading pre-trained model from {rfa_path}")
        d = torch.load(os.path.join(result_dir, suffix_load + '.pkl'), map_location='cpu')
        gen.load_state_dict(d)
    else:
        print(f"Pre-trained model {rfa_path} not found. Starting with a new model.")
    gen.to(device)
    gen.train()
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    writer_logdir = os.path.join(RFATexkwargs['writer_logdir'], TIMESTAMP + '_' + pargs.suffix)
    writer = SummaryWriter(logdir=writer_logdir)
    optimizerG = optim.Adam(gen.G.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    optimizerD = optim.Adam(gen.D.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))

    #yolov3
    epochs_v3 = RFATexkwargs['gen_epochs']
    filename = RFATexkwargs['gen_lossFile']

    for epoch in range(1, epochs_v3 + 1):
        with open(filename, 'a', newline='') as file:
            w = csv.writer(file)
            ep_ob_loss = 0
            ep_ob_loss_1 = 0
            ep_ob_loss_2 = 0
            ep_iou_loss = 0
            ep_de_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            D_loss = 0
            bt0 = time.time()
            test = 0
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader_v3), desc=f'Running epoch {epoch}',
                                                        total=epoch_length):

                img_batch = img_batch.to(device)
                lab_batch = lab_batch.to(device)
                # 随机向量
                z = torch.randn(img_batch.shape[0], args.z_dim, args.z_size, args.z_size, device=device)

                # 将随机向量输入到生成器中生成对抗纹理
                adv_patch = gen.generate(z)

                # tps变换
                adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5,
                                                 target_shape=adv_patch.shape[-2:])

                # 纹理增强
                adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
                                                pooling=args.pooling, old_fasion=kwargs['old_fasion'])

                # 将纹理贴到图片上
                p_img_batch = patch_applier(img_batch, adv_batch_t)


                # 计算损失
                ob_loss_1, ob_loss_2, iou_loss, valid_num = get_det_loss_v3(darknet_model_3, p_img_batch, lab_batch, args, kwargs, device)

                if valid_num > 0:
                    ob_loss_1 = ob_loss_1 / valid_num
                    ob_loss_2 = ob_loss_2 / valid_num
                    iou_loss = iou_loss / valid_num

                #加入Deterioration_Function的损失 关闭tvloss
                de_loss = Deterioration_Function(adv_patch)

                tv = total_variation(adv_patch)
                disc, pj, pm = gen.get_loss(adv_patch, z[:adv_patch.shape[0]], args.gp)
                # 可微变换损失
                tv_loss = tv * args.tv_loss
                # 训练稳定度
                disc_loss = disc * args.disc if epoch >= args.dim_start_epoch else disc * 0.0

                loss = ob_loss_1 + torch.max(tv_loss, torch.tensor(0.1).to(device))+ disc_loss +de_loss *0.1

                ep_ob_loss_1 += ob_loss_1.detach().item()
                ep_ob_loss_2 += ob_loss_2.detach().item()
                ep_iou_loss += iou_loss.detach().item()
                ep_de_loss +=de_loss.detach().item()
                ep_tv_loss += tv_loss.detach().item()
                ep_loss += loss.item()
                loss.backward()


                optimizerG.step()
                optimizerD.step()
                optimizerG.zero_grad()
                optimizerD.zero_grad()


                bt1 = time.time()
                if i_batch % 20 == 0:
                     iteration = epoch_length * epoch + i_batch

                     writer.add_scalar(Constants.TOTAL_LOSS, loss.item(), iteration)
                     writer.add_scalar(Constants.OB_LOSS_1, ob_loss_1.item(), iteration)
                     writer.add_scalar(Constants.OB_LOSS_2, ob_loss_2.item(), iteration)
                     writer.add_scalar(Constants.TV_LOSS, tv.item(), iteration)
                     writer.add_scalar(Constants.DE_LOSS, de_loss.item(), iteration)
                     writer.add_scalar(Constants.DISC_LOSS, disc.item(), iteration)
                     writer.add_scalar(Constants.DISC_PROB_TRUE, pj.mean().item(), iteration)
                     writer.add_scalar(Constants.DISC_PROB_FAKE, pm.mean().item(), iteration)
                     writer.add_scalar(Constants.MISC_EPOCH, epoch, iteration)
                     writer.add_scalar(Constants.MISC_LR, optimizerG.param_groups[0]["lr"], iteration)

                if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
                    writer.add_image('patch', adv_patch[0], iteration)
                    rpath = os.path.join(results_dir, 'patchv3%d' % epoch)
                    np.save(rpath, adv_patch.detach().cpu().numpy())
                    torch.save(gen.state_dict(), os.path.join(results_dir, pargs.suffix + 'v3.pkl'))

                bt0 = time.time()
            transforms.ToPILImage()(adv_patch[0].cpu()).save(os.path.join('..', 'outputs2', str(epoch) + '.png'))
            et1 = time.time()
            ep_det_loss = ep_ob_loss / len(loader_v3)
            ep_iou_loss = ep_iou_loss / len(loader_v3)
            ep_ob_loss_1 = ep_ob_loss_1 / len(loader_v3)
            ep_ob_loss_2 = ep_ob_loss_2 / len(loader_v3)
            ep_de_loss = ep_de_loss / len(loader_v3)
            w.writerow([epoch, ep_ob_loss_1, ep_ob_loss_2, ep_iou_loss , ep_de_loss])
            et0 = time.time()
    return gen


def train_RFA_Z(gen=None):
    if gen is None:
        gen = GAN_dis(DIM=args.DIM, z_dim=args.z_dim, img_shape=args.patch_size)
        suffix_load = RFATexkwargs['suffix_load']
        result_dir = RFATexkwargs['result_dir']
        rfa_path = os.path.join(result_dir, suffix_load + '.pkl')
        if os.path.exists(rfa_path):
            print(f"Loading pre-trained model from {rfa_path}")
            d = torch.load(os.path.join(result_dir, suffix_load + '.pkl'), map_location='cpu')
            gen.load_state_dict(d)
        else:
            print(f"Pre-trained model {rfa_path} not found. Starting with a new model.")
    gen.to(device)
    gen.eval()
    for p in gen.parameters():
        p.requires_grad = False

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    writer_logdir = os.path.join(RFATexkwargs['writer_logdir'], TIMESTAMP + '_' + pargs.suffix)
    writer = SummaryWriter(logdir=writer_logdir + '_z')

    # Generate stating point
    z0 = torch.randn(*args.z_shape, device=device)
    z = z0.detach().clone()
    z.requires_grad_(True)

    optimizer = optim.Adam([z], lr=args.learning_rate_z, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=500,
                                                     min_lr=args.learning_rate_z / 100)

    et0 = time.time()
    for epoch in range(1, args.z_epochs + 1):
        with open(RFATexkwargs['z_lossFile'], 'a', newline='') as file:
            w = csv.writer(file)

            ep_ob_loss = 0
            ep_ob_loss_1 = 0
            ep_ob_loss_2 = 0
            ep_iou_loss = 0
            ep_de_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader_v3), desc=f'Running epoch {epoch}',
                                                        total=epoch_length):
                img_batch = img_batch.to(device)
                lab_batch = lab_batch.to(device)
                z_crop, _, _ = random_crop(z, args.crop_size_z, pos=args.pos, crop_type=args.crop_type_z)

                adv_patch = gen.generate(z_crop)
                adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5,
                                                 target_shape=adv_patch.shape[-2:])
                adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
                                                pooling=args.pooling, old_fasion=kwargs['old_fasion'])
                p_img_batch = patch_applier(img_batch, adv_batch_t)


                # 计算损失
                ob_loss_1, ob_loss_2, iou_loss, valid_num = get_det_loss_v3(darknet_model_3, p_img_batch, lab_batch, args, kwargs, device)

                if valid_num > 0:
                    ob_loss_1 = ob_loss_1 / valid_num
                    ob_loss_2 = ob_loss_2 / valid_num
                    iou_loss = iou_loss / valid_num

                tv = total_variation(adv_patch)
                tv_loss = tv * args.tv_loss

                #计算Deterioration_Function损失 关闭tv损失
                de_loss = Deterioration_Function(adv_patch)
                loss = ob_loss_1 + torch.max(tv_loss, torch.tensor(0.1).to(device)) +de_loss *0.1
                ep_ob_loss_1 += ob_loss_1.detach().item()
                ep_ob_loss_2 += ob_loss_2.detach().item()
                ep_iou_loss += iou_loss.detach().item()
                ep_tv_loss += tv_loss.detach().item()
                ep_de_loss += de_loss.detach().item()
                ep_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                bt1 = time.time()
                if i_batch % 20 == 0:
                     iteration = epoch_length * epoch + i_batch

                     writer.add_scalar(Constants.TOTAL_LOSS, loss.detach().cpu().numpy(), iteration)
                     writer.add_scalar(Constants.OB_LOSS_1, ob_loss_1.detach().cpu().numpy(), iteration)
                     writer.add_scalar(Constants.OB_LOSS_2, ob_loss_2.detach().cpu().numpy(), iteration)
                     writer.add_scalar(Constants.TV_LOSS, tv.detach().cpu().numpy(), iteration)
                     writer.add_scalar(Constants.DE_LOSS, de_loss.detach().cpu().numpy(), iteration)
                     writer.add_scalar(Constants.MISC_EPOCH, epoch, iteration)
                     writer.add_scalar(Constants.MISC_LR, optimizer.param_groups[0]["lr"], iteration)

                if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
                    writer.add_image('patch', adv_patch.squeeze(0), iteration)
                    rpath = os.path.join(results_dir, 'patch%d' % epoch)
                    np.save(rpath, adv_patch.detach().cpu().numpy())
                    rpath = os.path.join(results_dir, 'z%d' % epoch)
                    np.save(rpath, z.detach().cpu().numpy())
                bt0 = time.time()
            transforms.ToPILImage()(adv_patch[0].cpu()).save(os.path.join('..', 'outputs3', str(epoch) + '.png'))
            et1 = time.time()
            ep_iou_loss = ep_iou_loss / len(loader_v3)
            ep_ob_loss_1 = ep_ob_loss_1 / len(loader_v3)
            ep_ob_loss_2 = ep_ob_loss_2 / len(loader_v3)
            ep_de_loss = ep_de_loss/ len(loader_v3)
            ep_tv_loss = ep_tv_loss /len(loader_v3)
            if epoch > 300:
                scheduler.step(ep_loss)
            w.writerow([epoch, ep_ob_loss_1, ep_ob_loss_2, ep_iou_loss])
    return 0

gen = train_RFA_GEN()
print('Start optimize z')
train_RFA_Z()



