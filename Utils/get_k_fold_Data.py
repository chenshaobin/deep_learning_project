"""
    # 将数据按照k折交叉验证进行划分
    # 最后将原始数据、每折训练和验证数据的图片和label写入到同一个excel文件的不同子表中
    # 本例假设有3种类别图像数据
"""
import re
from os.path import join
import os
import pandas as pd
import shutil
from typing import List
from collections import Counter
import random

class get_k_fold_Data:
    def __init__(self, excelFileSavePath: str, imgRootPath: str, kFold: int):
        """
        :param excelFileSavePath: 要存放的excel文件路径
        :param imgRootPath: 图片存放路径, 该路径下的图片已经按照文件夹分好类,文件夹名字首字母表示类别数字
        :param kFold: 划分数据的折数
        """
        self.imgRootPath = imgRootPath
        self.excelFileSavePath = excelFileSavePath
        self.kFold = kFold

    def get_k_fold_index(self, data: List, everyFoldSize: int, kthFold: int) -> List:
        """
        :param foldSize: 每折的数量
        :param kthFold: [0 , self.kFold - 1]
        :return: 返回对应的图片名称
        # 如果 (dataLength % kthFold) != 0,则除不尽多余的数据统一存放到最后一折数据中
        """
        assert kthFold <= self.kFold - 1, "输入的折数:{}超出范围!".format(kthFold)
        dataLength = len(data)
        train = []
        valid = []
        for j in range(self.kFold):
            idx = slice(j * everyFoldSize, (j + 1) * everyFoldSize)
            data_part = data[idx]
            if j == kthFold:
                # 属于验证集的那一折数据
                if kthFold == self.kFold - 1:
                    # 最后一折数据，在(dataLength % kthFold) != 0的情况下，将多余的数据也包含进去
                    index = slice(j * everyFoldSize, dataLength)
                    valid = data[index]
                else:
                    valid = data_part
            elif len(train) == 0:
                train = data_part
            elif j == (self.kFold - 1):
                # 最后一折数据，在(dataLength % kthFold) != 0的情况下，将多余的数据也包含进去
                index = slice(j * everyFoldSize, dataLength)
                data_part = data[index]
                train.extend(data_part)
            else:
                train.extend(data_part)

        return [train, valid]

    def createExcelFile(self, dataInfo: List, sheet_name: str):
        excelData = {'img': [item["img"] for item in dataInfo],
                     'label': [item["label"] for item in dataInfo]}
        df = pd.DataFrame(data=excelData)
        with pd.ExcelWriter(self.excelFileSavePath, mode='a', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name,  index=False)
        writer.save()
        writer.close()

    def getKFoldData(self):
        imgDataInfo = []    # 统计图像文件名及对应的label,格式：{'img':img1.tif, 'label': 0}
        for category in os.listdir(self.imgRootPath):
            categoryPath = join(self.imgRootPath, category)
            for img in os.listdir(categoryPath):
                imgInfo = {'img':img, 'label': category[0]}
                imgDataInfo.append(imgInfo)
        # 将数据信息写入到excel文件中
        df = pd.DataFrame(imgDataInfo)
        df.to_excel(excelFileSavePath, index=False, sheet_name='originalFile')
        category_0 = []
        category_1 = []
        category_2 = []
        for dataInfo in imgDataInfo:
            if dataInfo['label'] == '0':
                category_0.append(dataInfo)
            elif dataInfo['label'] == '1':
                category_1.append(dataInfo)
            elif dataInfo['label'] == '2':
                category_2.append(dataInfo)
            else:
                Exception("数据{}的类别异常，为{}:".format(dataInfo["label"], dataInfo["label"]))

        print('类别数据统计:0:{}, 1:{}, 2:{}'.format(len(category_0), len(category_1), len(category_2)))
        category0_foldSize = int(len(category_0) / self.kFold)
        category1_foldSize = int(len(category_1) / self.kFold)
        category2_foldSize = int(len(category_2) / self.kFold)
        print('category0_foldSize:{}, category1_foldSize:{}, category2_foldSize:{}, valid_foldSize:{}'
              .format(category0_foldSize, category1_foldSize, category2_foldSize, category0_foldSize + category1_foldSize + category2_foldSize))

        for kthFold in range(self.kFold):
            category_0_fold_train, category_0_fold_valid = self.get_k_fold_index(category_0, category0_foldSize, kthFold)
            print('category0 valid size:', len(category_0_fold_valid))
            assert len(category_0) == (len(category_0_fold_train) + len(category_0_fold_valid)), \
                '类别0,:原始数据量为:{}, 分折后的数据量为:{}'.format(len(category_0), len(category_0_fold_train) + len(category_0_fold_valid))

            category_1_fold_train, category_1_fold_valid = self.get_k_fold_index(category_1, category1_foldSize, kthFold)
            print('category1 valid size:', len(category_1_fold_valid))
            assert len(category_1) == (len(category_1_fold_train) + len(category_1_fold_valid)), \
                '类别1,:原始数据量为:{}, 分折后的数据量为:{}'.format(len(category_1), len(category_1_fold_train) + len(category_1_fold_valid))

            category_2_fold_train, category_2_fold_valid = self.get_k_fold_index(category_2, category2_foldSize, kthFold)
            print('category2 valid size:', len(category_2_fold_valid))
            assert len(category_2) == (len(category_2_fold_train) + len(category_2_fold_valid)), \
                '类别2,:原始数据量为:{}, 分折后的数据量为:{}'.format(len(category_2), len(category_2_fold_train) + len(category_2_fold_valid))

            fold_train = category_0_fold_train + category_1_fold_train + category_2_fold_train
            fold_valid = category_0_fold_valid + category_1_fold_valid + category_2_fold_valid
            print('fold:{} train size:{}'.format(kthFold, len(fold_train)))
            print('fold:{} valid size:{}'.format(kthFold, len(fold_valid)))
            assert len(imgDataInfo) == (len(fold_train) + len(fold_valid)),\
                '数据量不一致，请检查! --> 原始数据量为:{}, 分折后的数据量为:{}'.format(len(imgDataInfo), len(fold_train)+len(fold_valid))

            train_sheetName = 'train_' + 'fold' + str(kthFold)
            print('train sheetName:', train_sheetName)
            self.createExcelFile(fold_train, train_sheetName)

            valid_sheetName = 'valid_' + 'fold' + str(kthFold)
            print('valid sheetName:', valid_sheetName)
            self.createExcelFile(fold_valid, valid_sheetName)


if __name__ == '__main__':
    # 获取k折数据
    excelFileSavePath = r'D:\githubCode\Ethan_project\deep_learning_project\k_fold_data.xlsx'
    imgRootPath = r'D:\githubCode\Ethan_project\deep_learning_project\images'
    kFold = 5
    mainObj = get_k_fold_Data(excelFileSavePath=excelFileSavePath, imgRootPath=imgRootPath, kFold=kFold)
    mainObj.getKFoldData()
