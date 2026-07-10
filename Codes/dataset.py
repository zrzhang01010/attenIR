from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random
import torch.nn.functional as F


class TrainDataset(Dataset): #只用input图像，作为自编码式的训练数据
    # def __init__(self, data_path):

    #     self.width = 224
    #     self.height = 224 #训练数据图像全部是224*224
       
    #     self.train_path = data_path
        
    #     self.datas = OrderedDict()
        
    #     datas = glob.glob(os.path.join(self.train_path, '*'))
    #     for data in sorted(datas):
    #         data_name = data.split('/')[-1]
    #         if data_name == 'input':
    #             self.datas[data_name] = {}
    #             self.datas[data_name]['path'] = data
    #             self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
    #             self.datas[data_name]['image'].sort()
       

    # def __getitem__(self, index):
        
    #     # load image
        
    #     input = cv2.imread(self.datas['input']['image'][index])
    #     input = cv2.resize(input, (self.width, self.height))
    #     input = input.astype(dtype=np.float32) 
    #     input = (input / 127.5) - 1.0  
    #     input = np.transpose(input, [2, 0, 1])
       
    #     # convert to tensor
    #     input_tensor = torch.tensor(input)
        
    #     return input_tensor, input_tensor

    def __init__(
        self, data_path, saliency_dirname="saliency_train", use_saliency=True
    ):

        self.width = 224
        self.height = 224
       
        self.train_path = data_path
        self.use_saliency = use_saliency
        if os.path.isabs(saliency_dirname):
            self.saliency_root = saliency_dirname
        else:
            # data_root = /.../Data
            data_root = os.path.dirname(os.path.normpath(self.train_path))
            # saliency_root = /.../Data/saliency_train
            self.saliency_root = os.path.join(data_root, saliency_dirname)
        self._warned_missing_saliency = False
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()

    def __getitem__(self, index):
        
        # load image
        input_path = self.datas['input']['image'][index]
        input = cv2.imread(input_path)
        if input is None:
            raise ValueError(f"Failed to read input image: {input_path}")
        input = cv2.resize(input, (self.width, self.height))
        input = input.astype(dtype=np.float32)
        input = (input / 127.5) - 1.0
        input = np.transpose(input, [2, 0, 1])

        # load saliency map (IIM) aligned to input
        # if self.use_saliency:
        #     saliency_filename = os.path.basename(input_path)
        #     saliency_path = os.path.join(self.saliency_root, saliency_filename)
        #     iim = cv2.imread(saliency_path, cv2.IMREAD_GRAYSCALE)
        #     if iim is None:
        #         if not self._warned_missing_saliency:
        #             print(
        #                 f"Warning: missing saliency map, using ones. Example: {saliency_path}"
        #             )
        #             self._warned_missing_saliency = True
        #         iim = np.ones((self.height, self.width), dtype=np.float32)
        #     else:
        #         iim = cv2.resize(iim, (self.width, self.height))
        #         iim = iim.astype(dtype=np.float32) / 255.0
        #     iim = np.expand_dims(iim, axis=0)
        # else:
        #     iim = np.ones((1, self.height, self.width), dtype=np.float32)
        if self.use_saliency:
            saliency_filename = os.path.basename(input_path)
            saliency_path = os.path.join(self.saliency_root, saliency_filename)

            if (not os.path.isfile(saliency_path)):
                if not self._warned_missing_saliency:
                    print(f"Warning: missing saliency map, using ones. Example: {saliency_path}")
                    self._warned_missing_saliency = True
                iim = np.ones((self.height, self.width), dtype=np.float32)
            else:
                iim = cv2.imread(saliency_path, cv2.IMREAD_GRAYSCALE)
                if iim is None:
                    if not self._warned_missing_saliency:
                        print(f"Warning: saliency exists but unreadable, using ones. Example: {saliency_path}")
                        self._warned_missing_saliency = True
                    iim = np.ones((self.height, self.width), dtype=np.float32)
                else:
                    iim = cv2.resize(iim, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                    iim = iim.astype(np.float32) / 255.0

            iim = np.expand_dims(iim, axis=0)
        else:
            iim = np.ones((1, self.height, self.width), dtype=np.float32)

       
        # convert to tensor
        input_tensor = torch.tensor(input)
        iim_tensor = torch.tensor(iim)
      
        
        return input_tensor, input_tensor, iim_tensor    
       

    def __len__(self):

        return len(self.datas['input']['image'])#number of images in 'input'

class OutDataset(Dataset):
    """
    推理用数据集：
    - 默认返回 (input_tensor, input_path) 以便 test_output 用 basename 匹配显著性图
    - 可选 return_saliency=True 时，额外返回 iim_tensor: (input_tensor, input_path, iim_tensor)
    - 显著性图匹配策略：同 basename + suffix + ext（默认 suffix="_vis", ext 优先 png/jpg）
    """
    def __init__(
        self,
        data_path,
        use_saliency=False,
        saliency_dir="",
        saliency_suffix="_vis",   # 例如训练里是 basename+"_vis.png"
        saliency_exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
        resize_to=None,          # None 表示保持原始分辨率；(224,224) 表示推理前统一resize
    ):
        self.test_path = data_path
        self.use_saliency = use_saliency
        self.saliency_dir = saliency_dir
        self.saliency_suffix = saliency_suffix
        self.saliency_exts = saliency_exts
        self.resize_to = resize_to

        self.datas = OrderedDict()
        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.[jp][pn]g'))
                self.datas[data_name]['image'].sort()

        self._warned_missing_saliency = False

    def _find_saliency_file(saliency_dir, base_name,
                            exts=(".png",".jpg",".jpeg",".bmp",".tif",".tiff"),
                            suffixes=("", "_vis")):
        if not saliency_dir:
            return ""
        for suf in suffixes:
            for e in exts:
                p = os.path.join(saliency_dir, base_name + suf + e)
                if os.path.isfile(p):
                    return p
        return ""


    def __getitem__(self, index):
        input_path = self.datas['input']['image'][index]
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Failed to read input image: {input_path}")

        # 可选：推理前resize（若你的网络只稳定支持 224x224，可打开）
        if self.resize_to is not None:
            img = cv2.resize(img, self.resize_to, interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32)
        img = (img / 127.5) - 1.0
        img = np.transpose(img, [2, 0, 1])  # [C,H,W]
        input_tensor = torch.tensor(img)

        # 默认：返回路径给 test_output 做 basename 对齐
        if not self.use_saliency:
            return input_tensor, input_path

        # 可选：直接返回显著性张量
        H, W = input_tensor.shape[1], input_tensor.shape[2]
        sal_path = self._find_saliency_path(input_path)
        if not sal_path:
            if not self._warned_missing_saliency:
                print(f"[WARN] Missing saliency map. Example expected at: "
                      f"{os.path.join(self.saliency_dir, os.path.splitext(os.path.basename(input_path))[0] + self.saliency_suffix + self.saliency_exts[0])}")
                self._warned_missing_saliency = True
            iim = np.ones((H, W), dtype=np.float32)
        else:
            iim = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
            if iim is None:
                iim = np.ones((H, W), dtype=np.float32)
            else:
                iim = cv2.resize(iim, (W, H), interpolation=cv2.INTER_LINEAR)
                iim = iim.astype(np.float32) / 255.0

        iim = np.expand_dims(iim, axis=0)  # [1,H,W]
        iim_tensor = torch.tensor(iim, dtype=torch.float32)
        # return input_tensor, input_path, iim_tensor
        return input_tensor, self.datas['input']['image'][index]


    def __len__(self):
        return len(self.datas['input']['image'])

def collect_image_files(folder):
    """
    收集文件夹中的图像文件。
    """
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    image_files = []

    for ext in exts:
        image_files.extend(glob.glob(os.path.join(folder, ext)))
        image_files.extend(glob.glob(os.path.join(folder, ext.upper())))

    return sorted(image_files)


def get_image_stem(path):
    """
    获取不含扩展名的文件名，并统一转为小写。
    例如:
        /xxx/001.jpg -> 001
        /xxx/001.png -> 001
    """
    return os.path.splitext(os.path.basename(path))[0].lower()


def build_matched_image_pairs(
    input_path,
    output_path,
    output_suffixes=("", "_result", "_out", "_retarget", "_warp"),
    strict=False
):
    """
    按文件名匹配输入图像和输出图像。

    返回:
        matched_pairs: [(input_img_path, output_img_path), ...]
    """

    input_images = collect_image_files(input_path)
    output_images = collect_image_files(output_path)

    # 建立 output 文件名索引
    output_dict = {}

    for out_img in output_images:
        out_stem = get_image_stem(out_img)

        # 1. 完全同名匹配，例如 001.jpg -> 001.png
        if out_stem not in output_dict:
            output_dict[out_stem] = out_img

        # 2. 兼容带后缀的输出名，例如 001_result.png
        for suffix in output_suffixes:
            suffix = suffix.lower()
            if suffix != "" and out_stem.endswith(suffix):
                clean_stem = out_stem[:-len(suffix)]
                if clean_stem not in output_dict:
                    output_dict[clean_stem] = out_img

    matched_pairs = []
    missing_outputs = []

    for in_img in input_images:
        in_stem = get_image_stem(in_img)

        if in_stem in output_dict:
            matched_pairs.append((in_img, output_dict[in_stem]))
        else:
            missing_outputs.append(in_img)

    matched_output_set = set([pair[1] for pair in matched_pairs])
    extra_outputs = [
        out_img for out_img in output_images
        if out_img not in matched_output_set
    ]

    print("========== Matched Image Pairs ==========")
    print(f"Input images: {len(input_images)}")
    print(f"Output images: {len(output_images)}")
    print(f"Matched pairs: {len(matched_pairs)}")
    print(f"Missing outputs: {len(missing_outputs)}")
    print(f"Extra outputs: {len(extra_outputs)}")

    if len(missing_outputs) > 0:
        print("Examples of missing outputs:")
        for p in missing_outputs[:10]:
            print("  ", os.path.basename(p))

    if strict and len(missing_outputs) > 0:
        raise ValueError("Some input images do not have matched output images.")

    if len(matched_pairs) == 0:
        raise ValueError(
            "No matched input-output image pairs found. "
            "Please check filenames in input_path and output_path."
        )

    return matched_pairs

# class TestDataset(Dataset):
#     def __init__(self, data_path, ratio=0.5):
#         self.train_path = data_path
#         self.ratio = ratio  # Store the ratio parameter
#         self.datas = OrderedDict()
        
#         # 定义支持的图片后缀
#         img_extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'] 

#         datas = glob.glob(os.path.join(self.train_path, '*'))
#         for data in sorted(datas):
#             data_name = data.split('/')[-1]
#             if data_name == 'input' or data_name == 'output':
#                 self.datas[data_name] = {}
#                 self.datas[data_name]['path'] = data
                
#                 # 修改点：循环匹配多种后缀名
#                 image_list = []
#                 for ext in img_extensions:
#                     image_list.extend(glob.glob(os.path.join(data, ext)))
                
#                 # 排序确保 input 和 output 的顺序是一一对应的
#                 image_list.sort()
#                 self.datas[data_name]['image'] = image_list
                
#                 # 调试信息：打印看看找到了多少图片
#                 print(f"Successfully found {len(image_list)} images in {data_name} folder.")
       
#     def __getitem__(self, index):
#         # Load image
#         input_img = cv2.imread(self.datas['input']['image'][index])       
#         input_img = input_img.astype(dtype=np.float32)  
#         input_img = (input_img / 127.5) - 1.0
        
#         # Resize input to 640x640
#         input_img = cv2.resize(input_img, (640, 640), interpolation=cv2.INTER_LINEAR)
#         input_img = np.transpose(input_img, [2, 0, 1])  # Shape: [C, H, W]
        
#         output_img = cv2.imread(self.datas['output']['image'][index])        
#         output_img = output_img.astype(dtype=np.float32) 
#         output_img = (output_img / 127.5) - 1.0
        
#         # Resize output to width = 640 * ratio, height = 640
#         output_width = int(640 * self.ratio)
#         output_img = cv2.resize(output_img, (output_width, 640), interpolation=cv2.INTER_LINEAR)
#         output_img = np.transpose(output_img, [2, 0, 1])  # Shape: [C, H, W]
        
#         # Convert to tensor
#         input_tensor = torch.tensor(input_img)
#         output_tensor = torch.tensor(output_img)
        
#         return input_tensor, output_tensor
    
#     def __len__(self):
#         return len(self.datas['input']['image'])
       

#     def __len__(self):

#         return len(self.datas['input']['image'])

class TestDataset(Dataset):
    """
    用于 distortion error 测试的数据集。
    支持从外部传入已经按文件名匹配好的 matched_pairs。
    """

    def __init__(
        self,
        input_path,
        output_path,
        ratio=0.5,
        matched_pairs=None,
        image_size=640
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.ratio = ratio
        self.image_size = image_size

        if matched_pairs is None:
            self.matched_pairs = build_matched_image_pairs(
                input_path=self.input_path,
                output_path=self.output_path
            )
        else:
            self.matched_pairs = matched_pairs

    def _load_image(self, img_path, target_size):
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        img = img.astype(dtype=np.float32)
        img = (img / 127.5) - 1.0

        img = cv2.resize(
            img,
            target_size,
            interpolation=cv2.INTER_LINEAR
        )

        img = np.transpose(img, [2, 0, 1])

        return torch.tensor(img, dtype=torch.float32)

    def __getitem__(self, index):
        input_img_path, output_img_path = self.matched_pairs[index]

        input_tensor = self._load_image(
            input_img_path,
            target_size=(self.image_size, self.image_size)
        )

        output_width = max(1, int(round(self.image_size * self.ratio)))

        output_tensor = self._load_image(
            output_img_path,
            target_size=(output_width, self.image_size)
        )

        file_name = os.path.basename(input_img_path)

        return input_tensor, output_tensor, file_name

    def __len__(self):
        return len(self.matched_pairs)

class OutDataset(Dataset):
    def __init__(self, data_path):
        self.test_path = data_path
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.[jp][pn]g'))
                self.datas[data_name]['image'].sort()  # 按顺序排列图片路径

    def __getitem__(self, index):
        # 1. 获取当前索引对应的图片完整路径（关键：用于提取原文件名）
        img_full_path = self.datas['input']['image'][index]
        # 2. 提取纯文件名（如从 "/paddle/xxx/input/test001.jpg" 得到 "test001.jpg"）
        img_file_name = os.path.basename(img_full_path)
        
        # 原有逻辑：加载并预处理图片
        input_img = cv2.imread(img_full_path)
        input_img = input_img.astype(dtype=np.float32)
        input_img = (input_img / 127.5) - 1.0  # 归一化到 [-1, 1]
        input_img = np.transpose(input_img, [2, 0, 1])  # HWC -> CHW
        input_tensor = torch.tensor(input_img)
        
        # 关键修改：返回 (预处理后的tensor, 原文件名)，替代原来的 (input_tensor, input_tensor)
        return (input_tensor, img_file_name)

    def __len__(self):
        # 保持原有逻辑：返回input文件夹下的图片数量
        return len(self.datas['input']['image'])

def _collect_image_files(folder):
    """
    收集指定文件夹下的图像文件。
    支持 jpg / jpeg / png / bmp / tif / tiff。
    """
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    image_files = []

    for ext in exts:
        image_files.extend(glob.glob(os.path.join(folder, ext)))
        image_files.extend(glob.glob(os.path.join(folder, ext.upper())))

    return sorted(image_files)


def _image_stem(path):
    """
    获取不含扩展名的文件名，并统一转为小写。
    例如:
        /xxx/input/0001.jpg  -> 0001
        /xxx/output/0001.png -> 0001
    """
    return os.path.splitext(os.path.basename(path))[0].lower()


def build_matched_test_pairs(
    data_path,
    input_dirname="input",
    output_dirname="output",
    output_suffixes=None
):
    """
    在 data_path/input 和 data_path/output 中按文件名匹配图像。

    支持如下情况：
        input/ArtRoom.png
        output/ArtRoom_0.5.png

    此时会自动去掉 output 文件名中的 _0.5 后缀进行匹配。

    返回:
        matched_pairs: [(input_img_path, output_img_path), ...]
    """

    input_path = os.path.join(data_path, input_dirname)
    output_path = os.path.join(data_path, output_dirname)

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input folder not found: {input_path}")

    if not os.path.isdir(output_path):
        raise FileNotFoundError(f"Output folder not found: {output_path}")

    input_images = _collect_image_files(input_path)
    output_images = _collect_image_files(output_path)

    if len(input_images) == 0:
        raise ValueError(f"No input images found in: {input_path}")

    if len(output_images) == 0:
        raise ValueError(f"No output images found in: {output_path}")

    # 自动从 data_path 最后一级目录中提取比例后缀
    # 例如:
    # data_path = .../GPDM/0.5
    # ratio_name = 0.5
    # auto_ratio_suffix = _0.5
    ratio_name = os.path.basename(os.path.normpath(data_path))
    auto_ratio_suffix = "_" + ratio_name

    if output_suffixes is None:
        output_suffixes = (
            "",
            auto_ratio_suffix,
            "_0.5",
            "_0.75",
            "_1.25",
            "_1.5",
            "_1.75",
            "_result",
            "_out",
            "_retarget",
            "_warp"
        )

    # 建立 output 文件名索引
    output_dict = {}

    for out_img in output_images:
        out_stem = _image_stem(out_img)

        # 1. 完全同名匹配
        # input/ArtRoom.png -> output/ArtRoom.png
        if out_stem not in output_dict:
            output_dict[out_stem] = out_img

        # 2. 去除指定后缀后匹配
        # input/ArtRoom.png -> output/ArtRoom_0.5.png
        for suffix in output_suffixes:
            suffix = suffix.lower()

            if suffix != "" and out_stem.endswith(suffix):
                clean_stem = out_stem[:-len(suffix)]

                if clean_stem not in output_dict:
                    output_dict[clean_stem] = out_img

    matched_pairs = []
    missing_outputs = []

    for in_img in input_images:
        in_stem = _image_stem(in_img)

        if in_stem in output_dict:
            matched_pairs.append((in_img, output_dict[in_stem]))
        else:
            missing_outputs.append(in_img)

    matched_output_set = set([pair[1] for pair in matched_pairs])

    extra_outputs = [
        out_img for out_img in output_images
        if out_img not in matched_output_set
    ]

    print("========== Matched Test Images ==========")
    print(f"Test path: {data_path}")
    print(f"Input folder: {input_path}")
    print(f"Output folder: {output_path}")
    print(f"Input images: {len(input_images)}")
    print(f"Output images: {len(output_images)}")
    print(f"Matched pairs: {len(matched_pairs)}")
    print(f"Missing outputs: {len(missing_outputs)}")
    print(f"Extra outputs: {len(extra_outputs)}")
    print(f"Output suffixes: {output_suffixes}")

    if len(missing_outputs) > 0:
        print("Examples of missing outputs:")
        for p in missing_outputs[:10]:
            print("  ", os.path.basename(p))

    if len(extra_outputs) > 0:
        print("Examples of extra outputs:")
        for p in extra_outputs[:10]:
            print("  ", os.path.basename(p))

    if len(matched_pairs) == 0:
        raise ValueError(
            "No matched input-output pairs found. "
            "Please check whether filenames in input/ and output/ are consistent."
        )

    return matched_pairs


class TestDataset(Dataset):
    """
    用于 test_real2.py 的测试数据集。

    目录结构:
        data_path/input
        data_path/output

    匹配方式:
        按文件名 stem 匹配，而不是按排序 index 匹配。

    返回:
        input_tensor, output_tensor, file_name
    """

    def __init__(
        self,
        data_path,
        ratio=0.5,
        matched_pairs=None,
        image_size=640,
        input_dirname="input",
        output_dirname="output"
    ):
        self.data_path = data_path
        self.ratio = ratio
        self.image_size = image_size
        self.input_dirname = input_dirname
        self.output_dirname = output_dirname

        if matched_pairs is None:
            self.matched_pairs = build_matched_test_pairs(
                data_path=self.data_path,
                input_dirname=self.input_dirname,
                output_dirname=self.output_dirname
            )
        else:
            self.matched_pairs = matched_pairs

    def _load_image(self, img_path, target_size):
        """
        读取图像并转换为网络输入格式。

        target_size:
            OpenCV resize 使用的是 (width, height)
        """
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        img = cv2.resize(
            img,
            target_size,
            interpolation=cv2.INTER_LINEAR
        )

        img = img.astype(dtype=np.float32)
        img = (img / 127.5) - 1.0
        img = np.transpose(img, [2, 0, 1])

        return torch.tensor(img, dtype=torch.float32)

    def __getitem__(self, index):
        input_img_path, output_img_path = self.matched_pairs[index]

        # 输入统一 resize 到 640 x 640
        input_tensor = self._load_image(
            input_img_path,
            target_size=(self.image_size, self.image_size)
        )

        # 输出按照 ratio resize
        # ratio=0.5 时，输出为 320 x 640
        output_width = max(1, int(round(self.image_size * self.ratio)))

        output_tensor = self._load_image(
            output_img_path,
            target_size=(output_width, self.image_size)
        )

        file_name = os.path.basename(input_img_path)

        return input_tensor, output_tensor, file_name

    def __len__(self):
        return len(self.matched_pairs)

# class TestDataset(Dataset): # 带输入输出对的测试数据集，输入固定 640×640，输出为 640×(640*ratio)
#     def __init__(self, input_path, output_path, ratio=0.5):
#         self.input_path = input_path
#         self.output_path = output_path
#         self.ratio = ratio  # Store the ratio parameter
        
#         # 兼容 .jpg, .png, .jpeg
#         self.input_images = sorted(
#             glob.glob(os.path.join(self.input_path, '*.jpg')) + 
#             glob.glob(os.path.join(self.input_path, '*.png')) +
#             glob.glob(os.path.join(self.input_path, '*.jpeg'))
#         )
        
#         self.output_images = sorted(
#             glob.glob(os.path.join(self.output_path, '*.jpg')) + 
#             glob.glob(os.path.join(self.output_path, '*.png')) +
#             glob.glob(os.path.join(self.output_path, '*.jpeg'))
#         )
            
#         # 简单的防错检查：如果原图和结果图数量对不上，打印个提醒
#         if len(self.input_images) != len(self.output_images):
#             print(f"Warning: 输入图片数量 ({len(self.input_images)}) 和输出图片数量 ({len(self.output_images)}) 不一致，请检查文件夹！")

#     def __getitem__(self, index):
#         # Load image
#         input_img = cv2.imread(self.input_images[index])       
#         input_img = input_img.astype(dtype=np.float32)  
#         input_img = (input_img / 127.5) - 1.0
        
#         # Resize input to 640x640
#         input_img = cv2.resize(input_img, (640, 640), interpolation=cv2.INTER_LINEAR)
#         input_img = np.transpose(input_img, [2, 0, 1])  # Shape: [C, H, W]
        
#         output_img = cv2.imread(self.output_images[index])        
#         output_img = output_img.astype(dtype=np.float32) 
#         output_img = (output_img / 127.5) - 1.0
        
#         # Resize output to width = 640 * ratio, height = 640
#         output_width = int(640 * self.ratio)
#         output_img = cv2.resize(output_img, (output_width, 640), interpolation=cv2.INTER_LINEAR)
#         output_img = np.transpose(output_img, [2, 0, 1])  # Shape: [C, H, W]
        
#         # Convert to tensor
#         input_tensor = torch.tensor(input_img)
#         output_tensor = torch.tensor(output_img)
        
#         return input_tensor, output_tensor
    
#     def __len__(self):
#         return len(self.input_images)

# class TestDataset(Dataset): #带输入输出对的测试/训练数据集，输入固定 640×640，输出为 640×(640*ratio)
#     def __init__(self, data_path, ratio=0.5):
#         self.train_path = data_path
#         self.ratio = ratio  # Store the ratio parameter
#         self.datas = OrderedDict()
        
#         datas = glob.glob(os.path.join(self.train_path, '*'))
#         for data in sorted(datas):
#             data_name = data.split('/')[-1]
#             if data_name == 'input' or data_name == 'output':
#                 self.datas[data_name] = {}
#                 self.datas[data_name]['path'] = data
#                 self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
#                 self.datas[data_name]['image'].sort()
       
#     def __getitem__(self, index):
#         # Load image
#         input_img = cv2.imread(self.datas['input']['image'][index])       
#         input_img = input_img.astype(dtype=np.float32)  
#         input_img = (input_img / 127.5) - 1.0
        
#         # Resize input to 640x640
#         input_img = cv2.resize(input_img, (640, 640), interpolation=cv2.INTER_LINEAR)
#         input_img = np.transpose(input_img, [2, 0, 1])  # Shape: [C, H, W]
        
#         output_img = cv2.imread(self.datas['output']['image'][index])        
#         output_img = output_img.astype(dtype=np.float32) 
#         output_img = (output_img / 127.5) - 1.0
        
#         # Resize output to width = 640 * ratio, height = 640
#         output_width = int(640 * self.ratio)
#         output_img = cv2.resize(output_img, (output_width, 640), interpolation=cv2.INTER_LINEAR)
#         output_img = np.transpose(output_img, [2, 0, 1])  # Shape: [C, H, W]
        
#         # Convert to tensor
#         input_tensor = torch.tensor(input_img)
#         output_tensor = torch.tensor(output_img)
        
#         return input_tensor, output_tensor
    
#     def __len__(self):
#         return len(self.datas['input']['image'])
       

#     def __len__(self):

#         return len(self.datas['input']['image'])




def main():
    data_path = "/paddle/Zzr/Object-IR-saliency/Data/train"  
    dataset = TrainDataset(data_path)

    print(f"Total images in dataset: {len(dataset)}")

    failed_images = []
    for index in range(len(dataset)):
        try:
            _ = dataset[index] 
        except ValueError as e:
            print(e)  
            failed_images.append(str(e).split(": ")[-1])  


    if failed_images:
        print("\nFailed to load the following images:")
        for file in failed_images:
            print(file)
    else:
        print("\nAll images loaded successfully!")

if __name__ == "__main__":
    main()

