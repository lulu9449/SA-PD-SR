import cv2
import os
import numpy as np
import torch
import math


def info_log(log_file, message):
    print(message)
    log_file.write(message + "\n")
    # print("saved!")
    return


def save_tensor2imgs(imgs, dir_path, flag):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    b, c, h, w = imgs.shape
    imgs = imgs.detach().cpu().numpy()
    imgs = (imgs + 1.0) * 0.5 * 255.0
    imgs = np.array(np.clip(imgs, 0., 255.), dtype=np.uint8)
    imgs = imgs.transpose((0, 2, 3, 1))
    imgs = imgs[:, :, :, [2, 1, 0]]
    # print(dir_path, flag)
    for ind in range(b):
        img_name = "{}_{}.png".format(str(ind).zfill(4), flag)
        img_save_path = os.path.join(dir_path, img_name)
        img = imgs[ind]
        # cv2.imwrite(img_save_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(img_save_path, img)
        # print("INFO {} saved!".format(img_save_path))
    return


def save_kernel2imgs(kernels, dir_path, flag):
    bs, _, _, ho, wo, k_w, _ = kernels.shape
    kernels = torch.reshape(kernels, [bs, ho, wo, k_w, k_w])
    kernels = kernels.permute(0, 1, 3, 2, 4)
    kernels = torch.reshape(kernels, [bs, 1, ho*k_w, wo*k_w]).repeat(1, 3, 1, 1)
    # print(torch.max(kernels), torch.min(kernels))
    save_tensor2imgs(kernels * 2.0 - 1.0, dir_path, flag)
    return

def rgb2yuv(rgb):
    rgb_ = rgb.transpose(1, 3)
    A = torch.tensor([[0.299, -0.14714119, 0.61497538],
                      [0.587, -0.28886916, -0.51496512],
                      [0.114, 0.43601035, -0.10001026]]).cuda()
    # print(rgb_.shape, A.shape)
    yuv = torch.tensordot(rgb_ * 0.5 + 0.5, A, 1).transpose(1, 3)
    return yuv


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_psnr_ssim(img1, img2):
    return calculate_psnr(img1, img2), calculate_ssim(img1, img2)

