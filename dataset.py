from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torch.utils.data
import random
import os
from os.path import join, splitext, basename
from glob import glob
import pandas as pd
import math
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, Normalize, Resize, CenterCrop

def rand_crop(img, target_height, target_width):
    # reshape image to an appropriate size, and random crop to target size

    width = img.size[0]
    height = img.size[1]

    width_scale = target_width / width
    height_scale = target_height / height
    
    if height_scale >= 0.5:
        starting_x = random.randint(0, width - target_width)
        starting_y = random.randint(0, height - target_height)
    else:
        down_sample_ratio = height_scale / 0.5
        if round(down_sample_ratio*width) < target_width:
            down_sample_ratio = width_scale
        new_width = round(down_sample_ratio * width)
        new_height = round(down_sample_ratio * height)
        img = img.resize((new_width, new_height)) 
        starting_x = random.randint(0, new_width - target_width)
        starting_y = random.randint(0, new_height - target_height)
        
    img = img.crop((starting_x, starting_y, starting_x+target_width, starting_y+target_height))
    
    return img

## 2023 11 14 홍진성 사용
def center_crop(img, target_height, target_width):
    # reshape image to an appropriate size, and center crop to target size
    
    width = img.size[0]
    height = img.size[1]

    width_scale = target_width / width
    height_scale = target_height / height
    
    if height_scale >= 0.5:
        starting_x = (width - target_width) / 2
        starting_y = (height - target_height) / 2
    else:
        down_sample_ratio = height_scale / 0.5
        if round(down_sample_ratio*width) < target_width:
            down_sample_ratio = width_scale
        new_width = round(down_sample_ratio * width)
        new_height = round(down_sample_ratio * height)
        img = img.resize((new_width, new_height)) 
        starting_x = (new_width - target_width) / 2
        starting_y = (new_height - target_height) / 2
        
    img = img.crop((starting_x, starting_y, starting_x+target_width, starting_y+target_height))
    
    return img

class dataset_norm(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, imglist1=[], imglist2=[], imglist3=[]):  # 24.09.20 수정
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        #self.img_list = []
        self.transforms = transforms
        self.imgSize = imgSize
        self.inputsize = inputsize

        self.csvfile = '/content/drive/MyDrive/Recognition/Datasets/DS_TEST_bad(A)/registerds.csv'  # 24.10.10 HKDB-1

        self.img_list1 = imglist1
        self.img_list2 = imglist2
        self.img_list3 = imglist3

        # for name in file_list:
        #     img = Image.open(name)
        #     if (img.size[0] >= (self.imgSize)) and (img.size[1] >= self.imgSize):
        #         self.img_list += [name]

        self.size = len(self.img_list1)

    def image_name_change(self, img_name):
        parts = img_name.split('_')

        finger_part = parts[1]
        class_part = parts[2]
        new_class_part = f"{int(class_part)}"
        f_order_part = parts[3]

        new_filename = f"0_{new_class_part}_{f_order_part}_{finger_part}.bmp"

        return new_filename

    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region
        index = index % self.size

        # 24.09.20 수정. mathcing label과 img name 받아오기
        directory = self.img_list1[index]  # 학습에 사용하는 image의 경로 받아오기
        name_base = os.path.basename(directory)
        name_base = self.image_name_change(name_base)  # matching list와 다른 파일명 맞춰주기
        # print(f"Checking for file: {name_base}")

        csvfile = pd.read_csv(self.csvfile, header=None)  # matching이 쓰일 csv 파일 불러오기

        # print(csvfile)  # csv 파일은 잘 읽음

        final_list = []
        auth_matching_num = 1  # auth 매칭 개수
        impo_matching_num = 1  # impo 매칭 개수

        samelist = csvfile.iloc[:, 1].str.contains(name_base, na=False)  # 2열에서 searching

        # print(samelist)  # 부울 리스트로 잘 읽음

        if samelist.any():  # 2열 기준
            authlist = csvfile[samelist]  # 5개의 성분을 가진 Dataframe
            impolist = csvfile[~samelist]

            # 만약 authlist와 impolist가 비어있지 않으면 값을 추가
            if not authlist.empty:
                authlist_random = authlist.sample(n=auth_matching_num)  # auth 매칭 개수만큼 random으로 행 선택
                authlist_values = [[0, value] for value in authlist_random.iloc[:, 2].tolist()]  # auth label 0으로 추가, mathching dir
            else:
                print("authlist가 비어 있습니다.")

            if not impolist.empty:
                impolist_random = impolist.sample(n=impo_matching_num)  # impo 매칭 개수만큼 random으로 행 선택
                impolist_values = [[1, value] for value in impolist_random.iloc[:, 2].tolist()]  # impo label 1로 추가, mathching dir
            else:
                print("impolist가 비어 있습니다.")

            # 두 리스트를 합쳐서 최종 리스트 생성
            final_list = authlist_values + impolist_values

            if len(final_list) != (auth_matching_num + impo_matching_num):
                print(f"Warning: final_list의 크기가 2가 아닙니다. 현재 크기: {len(final_list)}")
                print(f"final_list 내용: {final_list}")

        if not samelist.any():  # 3열 기준
            # print(f"'{name_base}' not found in primary_column, searching in secondary_column instead.")

            #### 2th fold 일 때 클래스 부분 조정해줘야함!!!! 이외에는 0 ####
            class_add = 0  # HKdb-1, SDdb-1 기준
            # class_add = 156  # HKdb-2 기준
            # class_add = 318  # SDdb-2 기준

            class_base = str(int(directory.replace('\\', '/').split('/')[-2]) + class_add)  # 클래스 부분 추출
            classlist = csvfile.iloc[:, 2].str.contains(class_base, na=False)    # class 기준으로 찾기, 5개의 True
            # print(classlist)  # 잘 나옴

            if classlist.any():
                samelist = csvfile[classlist]  # class 기준 5개의 성분 list
                # print(samelist)  # 잘 나옴
                impolist = csvfile[~classlist]
                if not samelist.empty:
                    authlist = samelist.iloc[:, 2].str.contains(name_base, na=False)  # 1개의 성분을 가진 bool list,
                    authlist_random = samelist[authlist]  # 위와 변수 이름을 맞춰주기 위해 random 기입.
                    # print(authlist_random)
                    authlist_values = [[0, value] for value in authlist_random.iloc[:, 1].tolist()]
                    # print(authlist_values)
                if not impolist.empty:
                    impolist_random = impolist.sample(n=impo_matching_num)
                    impolist_values = [[1, value] for value in impolist_random.iloc[:, 1].tolist()]  # impolist에서 2열 값 추출
                    # print(impolist_values)

            else:
                print(f"'{class_base}'가 secondary_column에서도 찾을 수 없습니다.")

            # 두 리스트를 합쳐서 최종 리스트 생성
            final_list = authlist_values + impolist_values

            if len(final_list) != (auth_matching_num + impo_matching_num):
                print(f"Warning: final_list의 크기가 2가 아닙니다. 현재 크기: {len(final_list)}")
                print(f"final_list 내용: {final_list}")

        # 각 경로에서 이미지를 그레이스케일로 로드
        img = Image.open(self.img_list1[index]).convert("RGB")  # GT

        img1 = Image.open(self.img_list1[index]).convert("L")
        img2 = Image.open(self.img_list2[index]).convert("L")
        img3 = Image.open(self.img_list3[index]).convert("L")

        # 이미지를 numpy 배열로 변환
        img1 = np.array(img1)
        img2 = np.array(img2)
        img3 = np.array(img3)

        # 세 이미지 채널을 결합하여 RGB 이미지를 만듭니다.
        img_cat = np.stack([img1, img2, img3], axis=-1)  # (H, W, C) 형식으로 결합

        # 변환을 적용하기 위해 이미지를 PIL 이미지로 변환
        img_cat = Image.fromarray(img_cat)

        # 변환 적용
        if self.transforms:
            img = self.transforms(img)
            img_cat = self.transforms(img_cat)  # 자동으로 C이 index 0으로 옮겨짐

        i = (self.imgSize - self.inputsize)//2  # (192 - 128) / 2 = 32

        ## 2023 11 14 홍진성 마스킹 수정 (너비 방향으로만 처리)
        iner_img = img_cat[:, :, i:i+self.inputsize]
        mask_img = np.ones((3, self.imgSize, self.imgSize))
        mask_img[:, :, i:i + self.inputsize] = iner_img

        # # 텐서로 변환하여 저장
        # mask_img_tensor = torch.tensor(mask_img, dtype=torch.float32)
        #
        # # 저장할 경로 설정
        # save_dir = r'D:\DION4FR\output\HKdb-2\mask_images'  # 저장할 디렉토리 경로를 설정
        # os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성
        #
        # # 파일 이름 구성 및 저장
        # file_name = f'mask_image_{index}.png'
        # file_path = os.path.join(save_dir, file_name)
        # save_image(mask_img_tensor, file_path)

        return img, mask_img, final_list

    def __len__(self):
        return self.size



## 2023 11 14 홍진성 사용
class dataset_test4(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, pred_step=1, imglist=[]):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        self.preSize = imgSize + 64 * (pred_step - 1)
        self.inputsize = inputsize
        self.inputsize2 = inputsize + 64 * (pred_step - 1)

        self.img_list = imglist

        # for name in file_list:
        #     img = Image.open(name)
        #     #if (img.size[0] >= self.imgSize) and (img.size[1] >= self.imgSize):
        #     if (img.size[0] >= 0) and (img.size[1] >= 0):
        #         self.img_list += [name]

        self.size = len(self.img_list)
    ## 128 x 128 real 이고 나머지는 마스킹 영역
    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region

        index = index % self.size
        name = self.img_list[index]
        img = Image.open(name).convert('RGB')

        i = (self.imgSize - self.inputsize) // 2
        # j = (self.preSize - self.inputsize) // 2

        if self.transforms is not None:
            img = self.transforms(img)

        iner_img = img

        ## 2023 11 14 홍진성 마스킹 수정 (너비 방향으로만 처리)
        mask_img = np.ones((3, self.preSize, self.preSize))
        if self.pred_step > 1:
        #mask_img[:,i:i + self.inputsize + 64*(self.pred_step-1),i:i + self.inputsize + 32*self.pred_step]=img
            mask_img[:, i:i + self.inputsize2, i:i+self.inputsize2] = img
        else:
            mask_img[:, :, i:i + self.inputsize2] = iner_img

        return img, iner_img, mask_img, splitext(basename(name))[0], name.replace('\\', '/').split('/')[-2]
        # return torch.rand((3,192,192)), torch.rand((3,192,192)), torch.rand((3,192,192)), '', ''

    def __len__(self):
        return self.size
        # return 10000

class dataset_test3(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, crop='center', rand_pair=False, imgSize=192, inputsize=128):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        self.img_list = []
        self.transforms = transforms
        self.imgSize = imgSize
        self.rand_pair = rand_pair
        self.inputsize = inputsize
        if crop == 'rand':
            self.cropFunc = rand_crop
        elif crop == 'center':
            self.cropFunc = center_crop
        else:
            self.cropFunc = None

        file_list = sorted(glob(join(root, '*.jpg')))

        for name in file_list:
            img = Image.open(name)
            if (img.size[0] >= self.imgSize) and (img.size[1] >= self.imgSize):
                self.img_list += [name]

        self.size = len(self.img_list)

    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region

        index = index % self.size
        name = self.img_list[index]
        img = Image.open(name).convert('RGB')
        i = (self.imgSize - self.inputsize) // 2

        if self.cropFunc is not None:
            img = self.cropFunc(img, self.imgSize, self.imgSize)

        if self.transforms is not None:
            img = self.transforms(img)

        iner_img = img[:, i:i + self.inputsize, :]
        iner_img = iner_img[:, :, i:i+self.inputsize]
        mask_img = np.zeros((3, self.imgSize, self.imgSize))
        mask_img[:, i:i + self.inputsize, i:i + self.inputsize] = iner_img
        #mask[:, i:i + self.inputsize, i:i+self.inputsize] = 1
        #mask_img = img * mask
        #mask_img = np.ones((3, self.imgSize, self.imgSize))
        #mask_img[:, i:i + self.inputsize, i:i + self.inputsize] = iner_img

        return img, iner_img, mask_img, splitext(basename(name))[0]

    def __len__(self):
        return self.size

class dataset_arbi(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, pred_step=1):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        self.img_list = []
        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        self.preSize = imgSize + 64 * (pred_step - 1)
        self.inputsize = inputsize

        file_list = sorted(glob(join(root, '*.png')))

        for name in file_list:
            img = Image.open(name)
            # if (img.size[0] >= self.imgSize) and (img.size[1] >= self.imgSize):
            if (img.size[0] >= 0) and (img.size[1] >= 0):
                self.img_list += [name]

        self.size = len(self.img_list)

    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region

        index = index % self.size
        name = self.img_list[index]
        img = Image.open(name).convert('RGB')
        i = (self.imgSize - self.inputsize) // 2
        j = (self.preSize - self.inputsize) // 2

        if self.transforms is not None:
            img = self.transforms(img)

        #iner_img = img[:, i:i + self.inputsize, :]
        #iner_img = iner_img[:, :, i:i + self.inputsize]
        # mask_img = np.zeros((3, self.imgSize, self.imgSize))
        # mask[:, i:i + self.inputsize, i:i+self.inputsize] = 1
        # mask_img[:, i:i + self.inputsize, i:i+self.inputsize] = iner_img
        mask_img = np.ones((3, self.preSize, self.preSize))
        if self.pred_step > 1:
            # mask_img[:,i:i + self.inputsize + 64*(self.pred_step-1),i:i + self.inputsize + 32*self.pred_step]=img
            mask_img[:, i:i + self.imgSize, i:i + self.imgSize] = img
        else:
            mask_img[:, i:i + self.inputsize2, i:i + self.inputsize2] = img

        return img, img, mask_img, splitext(basename(name))[0]

    def __len__(self):
        return self.size

class dataset_arbi2(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, pred_step=1):

        self.img_list = []
        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        #self.preSize = imgSize + 64 * (pred_step - 1)
        self.inputsize = inputsize
        #self.cropsize = int(imgSize//(1.5**pred_step))

        self.img_list = sorted(glob(join(root, '*.jpg')))

        # for name in file_list:
        #     img = Image.open(name)
        #     # if (img.size[0] >= self.imgSize) and (img.size[1] >= self.imgSize):
        #     if (img.size[0] >= 0) and (img.size[1] >= 0):
        #         self.img_list += [name]

        self.size = len(self.img_list)

    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region

        index = index % self.size
        name = self.img_list[index]
        img = Image.open(name).convert('RGB')
        i = (self.imgSize - self.inputsize) // 2
        #j = (self.preSize - self.inputsize) // 2

        if self.transforms is not None:
            img = self.transforms(img)

        #crop = img[:,]

        iner_img = img[:, i:i + self.inputsize, :]
        iner_img = iner_img[:, :, i:i + self.inputsize]
        crop_img = Resize(128)(iner_img)
        # mask_img = np.zeros((3, self.imgSize, self.imgSize))
        # mask[:, i:i + self.inputsize, i:i+self.inputsize] = 1
        # mask_img[:, i:i + self.inputsize, i:i+self.inputsize] = iner_img
        mask_img = np.ones((3, self.imgSize, self.imgSize))
        mask_img[:,32:160,32:160] = crop_img

        return img, crop_img, mask_img, splitext(basename(name))[0]

    def __len__(self):
        return self.size

class dataset_arbi3(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, pred_step=1):

        self.img_list = []
        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        self.file_list = sorted(glob(join(root, '*.png')))

        self.size = len(self.file_list)

    def __getitem__(self, index):

        index = index % self.size
        name = self.file_list[index]
        img = Image.open(name).convert('RGB')


        if self.transforms is not None:
            img = self.transforms(img)

        mask_img = np.ones((3, self.imgSize, self.imgSize))
        mask_img[:,32:160,32:160] = img

        return img, img, mask_img, splitext(basename(name))[0]

    def __len__(self):
        return self.size

class dataset_arbi4(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, pred_step=1):

        self.img_list = []
        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        self.file_list = sorted(glob(join(root, '*.jpg')))

        self.size = len(self.file_list)

    def __getitem__(self, index):

        index = index % self.size
        name = self.file_list[index]
        img = Image.open(name)


        if self.transforms is not None:
            img = self.transforms(img)

        mask_img = np.ones((3, self.imgSize, self.imgSize))
        mask_img[:,32:160,32:160] = img

        return img, img, mask_img, splitext(basename(name))[0]

    def __len__(self):
        return self.size
