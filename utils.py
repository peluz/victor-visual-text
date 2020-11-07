from fastai.text.all import *
from fastai.vision.all import *
from sklearn.metrics import classification_report
import torch


def get_dls(path, bs, size):
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files,
                   get_y=parent_label,
                   splitter=GrandparentSplitter(valid_name="val"),
                   item_tfms=Resize(460),
                   batch_tfms=[*aug_transforms(size=size, min_scale=0.75,
                                               do_flip=False, max_rotate=0,
                                               max_warp=0
                                               ),
                               Normalize.from_stats(*imagenet_stats)])
    return dblock.dataloaders(path, bs=bs)

def evaluate(learn, dls, idx=1):
    preds, targets = learn.get_preds(dl=dls, ds_idx=idx)
    preds = np.argmax(preds, axis=1)
    print(classification_report(targets, preds, target_names=dls.vocab[-1], digits=4))
    
def evaluate_ensemble(fwd, bwd, fwd_dl, bwd_dl, idx=1):
    preds_fwd, targets = fwd.get_preds(dl=fwd_dl, ds_idx=idx)
    preds_bwd, _ = bwd.get_preds(dl=bwd_dl, ds_idx=idx)
    preds = (preds_fwd + preds_bwd)/2
    preds = np.argmax(preds, axis=1)
    print(classification_report(targets, preds, target_names=fwd.dl.vocab[-1], digits=4))