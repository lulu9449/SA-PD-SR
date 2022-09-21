import os
import cv2
from tqdm import tqdm

patch_size = 192
trainset_dirname = "/data/users/luluzhang_projects/datasets/DIV2K/DIV2K_train_HR"
save_patch_dirname = "/data/users/luluzhang_projects/datasets/DIV2K/DIV2K_train_HR_p"
if not os.path.exists(save_patch_dirname):
    os.mkdir(save_patch_dirname)
img_names = os.listdir(trainset_dirname)
for img_name in tqdm(img_names):
    img_path = os.path.join(trainset_dirname, img_name)
    img = cv2.imread(img_path)
    if not img.shape:
        print("{} is None!".format(img_path))
        continue
    h, w, _ = img.shape
    suffix = os.path.splitext(img_name)[-1]
    count = 0
    for i in range(0, h + 1 - patch_size, patch_size // 2):
        for j in range(0, w + 1 - patch_size, patch_size // 2):
            cur_patch = img[i:i + patch_size, j:j + patch_size]
            save_img_path = os.path.join(save_patch_dirname,
                                         img_name.replace(suffix, "_" + str(count).zfill(4) + ".png"))
            # print(cur_patch.shape)python
            cv2.imwrite(save_img_path,
                        cur_patch,
                        [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
                        )

            count += 1
        # if count > 9999:
        #     print("ERROR {} too large!".format(img_name))

