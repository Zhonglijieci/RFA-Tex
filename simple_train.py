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


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--net', default='yolov3', help='target net name')
    parser.add_argument('--method', default='RFA-Tex', help='method name')
    parser.add_argument('--suffix', default=None, help='suffix name')
    parser.add_argument('--gen_suffix', default=None, help='generator suffix name')
    parser.add_argument('--epoch', type=int, default=None, help='')
    parser.add_argument('--z_epoch', type=int, default=None, help='')
    parser.add_argument('--device', default='cuda:0', help='')
    return parser.parse_args()

def initialize_environment(pargs):
    args, kwargs = get_cfgs(pargs.net, pargs.method)
    RFATexkwargs = get_cfg(pargs.method)
    args.n_epochs = pargs.epoch if pargs.epoch is not None else args.n_epochs
    args.z_epochs = pargs.z_epoch if pargs.z_epoch is not None else args.z_epochs
    pargs.suffix = pargs.suffix if pargs.suffix is not None else f"{pargs.net}_{pargs.method}"
    device = torch.device(pargs.device)
    return args, kwargs, RFATexkwargs, device

def initialize_model(kwargs, device):
    model_path = kwargs['model_path']
    weights_path = kwargs['weights_path']
    darknet_model = load_yolo3(model_path, weights_path)
    darknet_model.eval().to(device)
    class_names = load_class_names(kwargs['classes_path'])
    return darknet_model, class_names

def setup_data_loaders(kwargs, args):
    train_data = load_data.InriaDataset(kwargs['img_dir_train'], kwargs['lab_dir_train_v3'], kwargs['max_lab'], args.img_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=kwargs['batch_size'], shuffle=True, num_workers=0)
    return train_loader

def prepare_directories(pargs, kwargs):
    results_dir = os.path.join(kwargs['result_dir'], pargs.suffix)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def setup_tps(device):
    target_control_points = torch.tensor(list(itertools.product(
        torch.arange(-1.0, 1.00001, 2.0 / 4),
        torch.arange(-1.0, 1.00001, 2.0 / 4),
    )))
    tps = TPSGridGen(torch.Size([300, 300]), target_control_points).to(device)
    return tps

def main():
    pargs = parse_arguments()
    args, kwargs, RFATexkwargs, device = initialize_environment(pargs)
    darknet_model, class_names = initialize_model(kwargs, device)
    train_loader = setup_data_loaders(kwargs, args)
    results_dir = prepare_directories(pargs, kwargs)
    tps = setup_tps(device)

    # 训练 RFA-Tex 模型
    gen = train_RFATex(darknet_model, train_loader, tps, device, args, kwargs, pargs, results_dir, RFATexkwargs)

    # 训练 z 模型
    train_z(gen, tps, device, args, kwargs, pargs, results_dir)

def train_RFATex(darknet_model, loader_v3, tps, device, args, kwargs, pargs, results_dir, RFATexkwargs):
    gen = GAN_dis(DIM=args.DIM, z_dim=args.z_dim, img_shape=args.patch_size)
    rfa_path = os.path.join(RFATexkwargs['result_dir'], RFATexkwargs['suffix_load'] + '.pkl')
    if os.path.exists(rfa_path):
        print(f"Loading pre-trained model from {rfa_path}")
        gen.load_state_dict(torch.load(rfa_path, map_location='cpu'))
    else:
        print(f"Pre-trained model {rfa_path} not found. Starting with a new model.")
    gen.to(device)
    gen.train()
    writer = SummaryWriter(logdir=os.path.join(RFATexkwargs['writer_logdir'], f"{datetime.now():%Y-%m-%dT%H-%M-%S}_{pargs.suffix}"))

    optimizerG = optim.Adam(gen.G.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    optimizerD = optim.Adam(gen.D.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))

    epochs_v3 = RFATexkwargs['epochs_v3']
    filename = RFATexkwargs['filename']

    for epoch in range(1, epochs_v3 + 1):
        with open(filename, 'a', newline='') as file:
            w = csv.writer(file)
            ep_ob_loss_1, ep_ob_loss_2, ep_iou_loss, ep_de_loss, ep_tv_loss, ep_loss = 0, 0, 0, 0, 0, 0

            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader_v3), desc=f'Running epoch {epoch}', total=len(loader_v3)):
                img_batch, lab_batch = img_batch.to(device), lab_batch.to(device)
                z = torch.randn(img_batch.shape[0], args.z_dim, args.z_size, args.z_size, device=device)
                adv_patch = gen.generate(z)

                adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
                adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False, pooling=args.pooling, old_fasion=kwargs['old_fasion'])
                p_img_batch = patch_applier(img_batch, adv_batch_t)

                ob_loss_1, ob_loss_2, iou_loss, valid_num = get_det_loss_v3(darknet_model, p_img_batch, lab_batch, args, kwargs, device)
                if valid_num > 0:
                    ob_loss_1, ob_loss_2, iou_loss = ob_loss_1 / valid_num, ob_loss_2 / valid_num, iou_loss / valid_num

                de_loss = Deterioration_Function(adv_patch)
                tv_loss = total_variation(adv_patch) * args.tv_loss
                disc, pj, pm = gen.get_loss(adv_patch, z[:adv_patch.shape[0]], args.gp)
                disc_loss = disc * args.disc if epoch >= args.dim_start_epoch else 0.0

                loss = ob_loss_1 * ob_loss_2 + torch.max(tv_loss, torch.tensor(0.1).to(device)) + disc_loss + de_loss * 0.1

                ep_ob_loss_1 += ob_loss_1.item()
                ep_ob_loss_2 += ob_loss_2.item()
                ep_iou_loss += iou_loss.item()
                ep_de_loss += de_loss.item()
                ep_tv_loss += tv_loss.item()
                ep_loss += loss.item()
                loss.backward()

                optimizerG.step()
                optimizerD.step()
                optimizerG.zero_grad()
                optimizerD.zero_grad()

                if i_batch % 20 == 0:
                    iteration = len(loader_v3) * epoch + i_batch
                    writer.add_scalar(Constants.TOTAL_LOSS, loss.item(), iteration)
                    writer.add_scalar(Constants.OB_LOSS_1, ob_loss_1.item(), iteration)
                    writer.add_scalar(Constants.OB_LOSS_2, ob_loss_2.item(), iteration)
                    writer.add_scalar(Constants.TV_LOSS, tv_loss.item(), iteration)
                    writer.add_scalar(Constants.DE_LOSS, de_loss.item(), iteration)
                    writer.add_scalar(Constants.DISC_LOSS, disc.item(), iteration)
                    writer.add_scalar(Constants.DISC_PROB_TRUE, pj.mean().item(), iteration)
                    writer.add_scalar(Constants.DISC_PROB_FAKE, pm.mean().item(), iteration)
                    writer.add_scalar(Constants.MISC_EPOCH, epoch, iteration)
                    writer.add_scalar(Constants.MISC_LR, optimizerG.param_groups[0]["lr"], iteration)

                if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
                    writer.add_image('patch', adv_patch[0], iteration)
                    np.save(os.path.join(results_dir, f'patchv3{epoch}.npy'), adv_patch.detach().cpu().numpy())
                    torch.save(gen.state_dict(), os.path.join(results_dir, f'{pargs.suffix}v3.pkl'))

            transforms.ToPILImage()(adv_patch[0].cpu()).save(os.path.join('..', 'outputs2', f'{epoch}.png'))
            w.writerow([epoch, ep_ob_loss_1 / len(loader_v3), ep_ob_loss_2 / len(loader_v3), ep_iou_loss / len(loader_v3), ep_de_loss / len(loader_v3)])

    return gen

def train_z(gen, tps, device, args, kwargs, pargs, results_dir):
    if gen is None:
        gen = GAN_dis(DIM=args.DIM, z_dim=args.z_dim, img_shape=args.patch_size)
        rfa_path = os.path.join(RFATexkwargs['result_dir'], RFATexkwargs['suffix_load'] + '.pkl')
        if os.path.exists(rfa_path):
            print(f"Loading pre-trained model from {rfa_path}")
            gen.load_state_dict(torch.load(rfa_path, map_location='cpu'))
        else:
            print(f"Pre-trained model {rfa_path} not found. Starting with a new model.")
    gen.to(device)
    gen.eval()
    for p in gen.parameters():
        p.requires_grad = False

    writer = SummaryWriter(logdir=os.path.join(RFATexkwargs['writer_logdir'], f"{datetime.now():%Y-%m-%dT%H-%M-%S}_{pargs.suffix}_z"))

    z = torch.randn(*args.z_shape, device=device, requires_grad=True)

    optimizer = optim.Adam([z], lr=args.learning_rate_z, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=500, min_lr=args.learning_rate_z / 100)

    filename = 'z_loss.csv'
    for epoch in range(1, args.z_epochs + 1):
        with open(filename, 'a', newline='') as file:
            w = csv.writer(file)

            ep_ob_loss_1, ep_ob_loss_2, ep_iou_loss, ep_de_loss, ep_tv_loss, ep_loss = 0, 0, 0, 0, 0, 0

            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader_v3), desc=f'Running epoch {epoch}', total=epoch_length):
                img_batch = img_batch.to(device)
                lab_batch = lab_batch.to(device)
                z_crop, _, _ = random_crop(z, args.crop_size_z, pos=args.pos, crop_type=args.crop_type_z)

                adv_patch = gen.generate(z_crop)
                adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
                adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False, pooling=args.pooling, old_fasion=kwargs['old_fasion'])
                p_img_batch = patch_applier(img_batch, adv_batch_t)

                ob_loss_1, ob_loss_2, iou_loss, valid_num = get_det_loss_v3(darknet_model, p_img_batch, lab_batch, args, kwargs, device)
                if valid_num > 0:
                    ob_loss_1, ob_loss_2, iou_loss = ob_loss_1 / valid_num, ob_loss_2 / valid_num, iou_loss / valid_num

                tv_loss = total_variation(adv_patch) * args.tv_loss
                de_loss = Deterioration_Function(adv_patch)
                loss = ob_loss_1 * ob_loss_2 + torch.max(tv_loss, torch.tensor(0.1).to(device)) + de_loss * 0.1

                ep_ob_loss_1 += ob_loss_1.detach().item()
                ep_ob_loss_2 += ob_loss_2.detach().item()
                ep_iou_loss += iou_loss.detach().item()
                ep_tv_loss += tv_loss.detach().item()
                ep_de_loss += de_loss.detach().item()
                ep_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if i_batch % 20 == 0:
                    iteration = len(loader_v3) * epoch + i_batch
                    writer.add_scalar(Constants.TOTAL_LOSS, loss.item(), iteration)
                    writer.add_scalar(Constants.OB_LOSS_1, ob_loss_1.item(), iteration)
                    writer.add_scalar(Constants.OB_LOSS_2, ob_loss_2.item(), iteration)
                    writer.add_scalar(Constants.TV_LOSS, tv_loss.item(), iteration)
                    writer.add_scalar(Constants.DE_LOSS, de_loss.item(), iteration)
                    writer.add_scalar(Constants.MISC_EPOCH, epoch, iteration)
                    writer.add_scalar(Constants.MISC_LR, optimizer.param_groups[0]["lr"], iteration)

                if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
                    writer.add_image('patch', adv_patch.squeeze(0), iteration)
                    np.save(os.path.join(results_dir, f'patch{epoch}.npy'), adv_patch.detach().cpu().numpy())
                    np.save(os.path.join(results_dir, f'z{epoch}.npy'), z.detach().cpu().numpy())

            transforms.ToPILImage()(adv_patch[0].cpu()).save(os.path.join('..', 'outputs3', f'{epoch}.png'))
            w.writerow([epoch, ep_ob_loss_1 / len(loader_v3), ep_ob_loss_2 / len(loader_v3), ep_iou_loss / len(loader_v3), ep_de_loss / len(loader_v3)])

            if epoch > 300:
                scheduler.step(ep_loss)

if __name__ == "__main__":
    main()



