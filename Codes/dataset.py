from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random
import torch.nn.functional as F


class TrainDataset(Dataset):
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

    def __init__(
        self,
        data_path,
        use_saliency=False,
        saliency_dir="",
        saliency_suffix="_vis",   
        saliency_exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
        resize_to=None,         
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


        if self.resize_to is not None:
            img = cv2.resize(img, self.resize_to, interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32)
        img = (img / 127.5) - 1.0
        img = np.transpose(img, [2, 0, 1])  # [C,H,W]
        input_tensor = torch.tensor(img)

        if not self.use_saliency:
            return input_tensor, input_path

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
   
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    image_files = []

    for ext in exts:
        image_files.extend(glob.glob(os.path.join(folder, ext)))
        image_files.extend(glob.glob(os.path.join(folder, ext.upper())))

    return sorted(image_files)


def get_image_stem(path):
    
    return os.path.splitext(os.path.basename(path))[0].lower()


def build_matched_image_pairs(
    input_path,
    output_path,
    output_suffixes=("", "_result", "_out", "_retarget", "_warp"),
    strict=False
):
    

    input_images = collect_image_files(input_path)
    output_images = collect_image_files(output_path)

 
    output_dict = {}

    for out_img in output_images:
        out_stem = get_image_stem(out_img)

     
        if out_stem not in output_dict:
            output_dict[out_stem] = out_img

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
                self.datas[data_name]['image'].sort()  

    def __getitem__(self, index):

        img_full_path = self.datas['input']['image'][index]
       
        img_file_name = os.path.basename(img_full_path)
        

        input_img = cv2.imread(img_full_path)
        input_img = input_img.astype(dtype=np.float32)
        input_img = (input_img / 127.5) - 1.0  
        input_img = np.transpose(input_img, [2, 0, 1])  
        input_tensor = torch.tensor(input_img)
        

        return (input_tensor, img_file_name)

    def __len__(self):

        return len(self.datas['input']['image'])

def _collect_image_files(folder):
   
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    image_files = []

    for ext in exts:
        image_files.extend(glob.glob(os.path.join(folder, ext)))
        image_files.extend(glob.glob(os.path.join(folder, ext.upper())))

    return sorted(image_files)


def _image_stem(path):
   
    return os.path.splitext(os.path.basename(path))[0].lower()


def build_matched_test_pairs(
    data_path,
    input_dirname="input",
    output_dirname="output",
    output_suffixes=None
):
    

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

    output_dict = {}

    for out_img in output_images:
        out_stem = _image_stem(out_img)

        if out_stem not in output_dict:
            output_dict[out_stem] = out_img

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



def main():
    data_path = ""  
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

