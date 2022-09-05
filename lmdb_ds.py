import numpy as np
import os
import os.path as osp
import pickle
import compress_pickle
import torch
from PIL.Image import Image
from torch.utils import data
# from compress_pickle import dumps, loads
from lmdb_image import LMDB_Image
from  seg_code_2d.seg_code_2d import data_loader_sn
from utils_data_proc import *
import lmdb

def lmdb2lmdb():
    db_path_read = r'/data/medai05/PCa/train2ch.lmdb'
    db_path_save =r'/data/medai05/PCa/train2ch_compressed_gzip.lmdb'

    env = lmdb.open(db_path_read, subdir=os.path.isdir(db_path_read),
                         readonly=True, lock=False,
                         readahead=False, meminit=False)
    isdir = os.path.isdir(db_path_save)


    db = lmdb.open(db_path_save, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)
    with env.begin() as txn_read:
        length = pickle.loads(txn_read.get(b'__len__'))
        keys = pickle.loads(txn_read.get(b'__keys__'))
        idx=1
        for key in keys:
            byteflow = txn_read.get(key)
            IMAGE = pickle.loads(byteflow)
            img = IMAGE.get_image()

            temp = LMDB_Image(img)
            txn.put(key, compress_pickle.dumps(temp,compression='gzip'))
            print(key)
            if idx % 100 == 0:
                print(key)
                print("[%d/%d]" % (idx, len(keys)))
                txn.commit()
                txn = db.begin(write=True)
            idx+=1
        txn.commit()
        # keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
        with db.begin(write=True) as txn:
            txn.put(b'__keys__', pickle.dumps(keys))
            txn.put(b'__len__', pickle.dumps(len(keys)))

        print("Flushing database ...")
        db.sync()
        db.close()

lmdb2lmdb()


def data2lmdb(dpath, name="train", write_frequency=1000, num_workers=4):
    # 获取自定义的COCO数据集（就是最原始的那个直接从磁盘读取image的数据集）
    save_folder = r'/data/newnas/ZSN/2022_miccai_petct/data/h5_data/v1/'
    # all_files_list = [osp.join(save_folder,'0.0.h5'),osp.join(save_folder,'0.1.h5'),osp.join(save_folder,'0.2.h5')]
    all_files_list = os.listdir(save_folder)
    all_files_list = [osp.join(save_folder,f) for f in all_files_list]

    dataset=data_loader_sn.ImageFolder_cvt_lmdb(h5list=all_files_list)
    # data_loader = DataLoader(dataset, num_workers=8, collate_fn=lambda x: x)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,collate_fn=lambda x: x)
    lmdb_path = osp.join(dpath,"%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)
    keys = []
    for idx, data in enumerate(data_loader):
        image,pid,lid= data[0]
        image = image.astype(np.float16)
        keys.append((str(pid)+'.'+str(lid)).encode('ascii'))
        temp = LMDB_Image(image)
        txn.put(keys[idx], pickle.dumps(temp))
        print(keys[idx])
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    # keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', pickle.dumps(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

# data2lmdb(r'/data/newnas/ZSN/2022_miccai_petct/data/lmdb','train5ch')

class DatasetLMDB(data.Dataset):
    def __init__(self, db_path, transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length =pickle.loads(txn.get(b'__len__'))
            self.keys= pickle.loads(txn.get(b'__keys__'))
            # self.NP= pickle.loads(txn.get(b'__NP__'))
        # self.class_weights=torch.load("/data/jxzhang/coco/data/classweight.pt")
        self.transform = transform
        self.num_classes=80

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin() as txn:
            byteflow = txn.get(self.keys[index])
        IMAGE= pickle.loads(byteflow)
        img, label = IMAGE.get_image(),IMAGE.label
        return Image.fromarray(img).convert('RGB'),label.copy()

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'