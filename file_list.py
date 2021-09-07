# Image loaders for images and lables must be listed in a separate file
# see https://github.com/pytorch/vision/issues/81

from PIL import Image
import torch.utils.data as data
import os


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.targets = [x[1] for x in self.imlist]

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)


class ImageAnnotationfiles(data.Dataset):
    def __init__(self, root, flist, annotations_dir, images_dir, transform=None, target_transform=None,
                 annotations_reader=None, loader=default_loader):
        self.root = root
        self.imlist = annotations_reader(root, flist, annotations_dir, images_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.targets = [x[1] for x in self.imlist]

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)

