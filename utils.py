import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from yolo2 import utils
from pytorchyolo.utils.utils import non_max_suppression

def truths_length(truths):
    for i in range(len(truths)):
        if truths[i][1] == -1:
            return i
    return len(truths)

def get_det_loss_v3(darknet_model, p_img, lab_batch, args, kwargs, device):

    # 初始化
    valid_num = 0
    det_loss_1 = p_img.new_zeros([])
    det_loss_2 = p_img.new_zeros([])
    iou_loss = p_img.new_zeros([])

    for ii in range(p_img.shape[0]):
        # with torch.no_grad():
            # detections = darknet_model(p_img[ii].unsqueeze(0))

        # 把图片输入到yolov3中，不是完全得到结果，得到的是特征图结果
        detections = darknet_model(p_img[ii].unsqueeze(0))

        # 结果，输出识别框
        all_boxes = non_max_suppression(detections, conf_thres=0.5)[0]
        # all_boxes.requires_grad_(True)

        if all_boxes.shape[0] > 0:
            torch.cuda.empty_cache()

            boxes, m = deal_box(all_boxes)
            boxes = boxes.to(device)
            # 对比识别框和标签数据框,得到重合度
            iou_mat = utils.bbox_iou_mat(boxes, lab_batch[ii][:truths_length(lab_batch[ii]), 1:],
                                         False)
            iou_max = iou_mat.max(1)[0]
            # 识别框重合度阈值
            idxs = iou_max > args.iou_thresh
            # det_confs = m[idxs][:, 4]
            # 修改：置信度相乘概率
            det_m_1 = m[idxs][:, 4]
            det_m_2 = m[idxs][:, 6]
            mul_det_confs = m[idxs][:, 4] * m[idxs][:, 6]
            if mul_det_confs.shape[0] > 0:
                # 修改：用相乘结果
                # max_prob = mul_det_confs.max()
                # max_prob = det_confs.max()
                # ***
                # 识别狂重合度loss
                iou = iou_max[iou_max > args.iou_thresh]
                max_iou = torch.argmax(mul_det_confs)
                iou_loss = iou_loss + iou[max_iou]
                # +++

                # 修改：找到相乘结果最大的
                det_loss_1 = det_loss_1 + det_m_1[max_iou]
                det_loss_2 = det_loss_2 + det_m_2[max_iou]
                # ***

                valid_num += 1

    return det_loss_1, det_loss_2, iou_loss, valid_num

def deal_box(boxes):
    t = boxes[..., :4]
    num = 0
    # 这里只保留识别了类别=0（人）的数据
    for i in range(t.shape[0]):
        if(boxes[i][5]==0):
            num = num + 1
    k = 0
    box = torch.zeros(num, 4)
    # 修改：将置信度加入
    m = torch.zeros(num, 7)
    # m = torch.zeros(num, 6)
    for i in range(t.shape[0]):
        if(boxes[i][5]==0):
            box[k] = t[i]
            m[k] = boxes[i]
            k = k + 1
    t = box
    for i in range(t.shape[0]):
        w1 = t[i][2] - t[i][0]
        h1 = t[i][3] - t[i][1]
        w = w1 / 416
        h = h1 / 416
        s1 = t[i][0] + w1/2
        s2 = t[i][1] + h1/2
        x1 = s1 / 416
        x2 = s2 / 416
        t[i][0] = x1
        t[i][1] = x2
        t[i][2] = w
        t[i][3] = h

    return t, m

def get_det_loss_v2(darknet_model, p_img, lab_batch, args, kwargs):
    valid_num = 0
    det_loss = p_img.new_zeros([])
    output = darknet_model(p_img)
    if kwargs['name'] == 'yolov2':
        all_boxes_t = [utils.get_region_boxes_general(output, darknet_model, conf_thresh=args.conf_thresh, name=kwargs['name'])]
    else:
        raise ValueError

    for all_boxes in all_boxes_t:
        for ii in range(p_img.shape[0]):
            s = all_boxes[ii]
            if all_boxes[ii].shape[0] > 0:
                iou_mat = utils.bbox_iou_mat(all_boxes[ii][..., :4], lab_batch[ii][:truths_length(lab_batch[ii]), 1:], False)
                iou_max = iou_mat.max(1)[0]
                idxs = iou_max > args.iou_thresh
                det_confs = all_boxes[ii][idxs][:, 4]
                if det_confs.shape[0] > 0:
                    max_prob = det_confs.max()
                    det_loss = det_loss + max_prob
                    valid_num += 1

    return det_loss, valid_num

def gauss_kernel(ksize=5, sigma=None, conv=False, dtype=np.float32):
    half = (ksize - 1) * 0.5
    if sigma is None:
        sigma = 0.3 * (half - 1) + 0.8
    x = np.arange(-half, half + 1)
    x = np.exp(- np.square(x / sigma) / 2)
    x = np.outer(x, x)
    x = x / x.sum()
    if conv:
        kernel = np.zeros((3, 3, ksize, ksize))
        for i in range(3):
            kernel[i, i] = x
    else:
        kernel = x
    return kernel.astype(dtype)


def pad_and_scale(img, lab=None, size=(416, 416), color=(127, 127, 127)):
    """

    Args:
        img:

    Returns:

    """
    w, h = img.size
    if w == h:
        padded_img = img
    else:
        dim_to_pad = 1 if w < h else 2
        if dim_to_pad == 1:
            padding = (h - w) / 2
            padded_img = Image.new('RGB', (h, h), color=color)
            padded_img.paste(img, (int(padding), 0))
            if lab is not None:
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
        else:
            padding = (w - h) / 2
            padded_img = Image.new('RGB', (w, w), color=color)
            padded_img.paste(img, (0, int(padding)))
            if lab is not None:
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h / w)
    padded_img = padded_img.resize((size[0], size[1]))
    if lab is None:
        return padded_img
    else:
        return padded_img, lab


def random_crop(cloth, crop_size, pos=None, crop_type=None, fill=0):
    w = cloth.shape[2]
    h = cloth.shape[3]
    if crop_size is 'equal':
        crop_size = [w, h]
    if crop_type is None:
        d_w = w - crop_size[0]
        d_h = h - crop_size[1]
        if pos is None:
            r_w = np.random.randint(d_w + 1)
            r_h = np.random.randint(d_h + 1)
        elif pos == 'center':
            r_w, r_h = (np.array(cloth.shape[2:]) - np.array(crop_size)) // 2
        else:
            r_w = pos[0]
            r_h = pos[1]

        p1 = max(0, 0 - r_h)
        p2 = max(0, r_h + crop_size[1] - h)
        p3 = max(0, 0 - r_w)
        p4 = max(0, r_w + crop_size[1] - w)
        cloth_pad = F.pad(cloth, [p1, p2, p3, p4], value=fill)
        patch = cloth_pad[:, :, r_w:r_w + crop_size[0], r_h:r_h + crop_size[1]]

    elif crop_type == 'recursive':
        if pos is None:
            r_w = np.random.randint(w)
            r_h = np.random.randint(h)
        elif pos == 'center':
            r_w, r_h = (np.array(cloth.shape[2:]) - np.array(crop_size)) // 2
            if r_w < 0:
                r_w = r_w % w
            if r_h < 0:
                r_h = r_h % h
        else:
            r_w = pos[0]
            r_h = pos[1]
        expand_w = (w + crop_size[0] - 1) // w + 1
        expand_h = (h + crop_size[1] - 1) // h + 1
        cloth_expanded = cloth.repeat([1, 1, expand_w, expand_h])
        patch = cloth_expanded[:, :, r_w:r_w + crop_size[0], r_h:r_h + crop_size[1]]

    else:
        raise ValueError
    return patch, r_w, r_h


def random_stick(inputs, patch, stick_size=None, mode='replace', pos=None):
    if stick_size is None:
        stick_size = patch.shape[2:4]
    w = inputs.shape[2]
    h = inputs.shape[3]
    d_w = w - stick_size[0]
    d_h = h - stick_size[1]
    if pos is None:
        r_w = np.random.randint(d_w + 1)
        r_h = np.random.randint(d_h + 1)
    elif pos == 'center':
        r_w, r_h = (np.array(inputs.shape[2:]) - np.array(stick_size)) // 2
    else:
        r_w = pos[0]
        r_h = pos[1]

    patch_stick = inputs.new_zeros(inputs.shape)
    patch_resized = patch
    patch_stick[:, :, r_w:r_w + stick_size[0], r_h:r_h + stick_size[1]] = patch_resized

    assert mode in ['add', 'replace']

    if mode == 'add':
        inputs_stick = (inputs + patch_stick).clamp(0, 1)
    #         return inputs_stick

    elif mode == 'replace':
        mask = inputs.new_zeros(inputs.shape)
        mask[:, :, r_w:r_w + stick_size[0], r_h:r_h + stick_size[1]] = 1
        inputs_stick = mask * patch_stick + (1 - mask) * inputs
        inputs_stick = inputs_stick.clamp(0, 1)
    else:
        inputs_stick = None

    return inputs_stick, r_w, r_h


def TVLoss(patch):

    t1 = (patch[:, :, 1:, :] - patch[:, :, :-1, :]).abs().sum()
    t2 = (patch[:, :, :, 1:] - patch[:, :, :, :-1]).abs().sum()

    tv = t1 + t2

    return tv / patch.numel()

