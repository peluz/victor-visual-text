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

def evaluate(learn, dls):
    preds, targets = learn.get_preds(dl=dls)
    preds = np.argmax(preds, axis=1)
    print(classification_report(targets, preds, target_names=dls.vocab[-1], digits=4))
    
def evaluate_ensemble(fwd, bwd, fwd_dl, bwd_dl):
    preds_fwd, targets = fwd.get_preds(dl=fwd_dl)
    preds_bwd, _ = bwd.get_preds(dl=bwd_dl)
    preds = (preds_fwd + preds_bwd)/2
    preds = np.argmax(preds, axis=1)
    print(classification_report(targets, preds, target_names=fwd.dl.vocab[-1], digits=4))
    
def get_sequences(data):
    xs = []
    ys = []
    for k, v in data.groupby("process_id").groups.items():
        xs.append(data.iloc[v]["activation_path"].tolist())
        ys.append(data.iloc[v]["document_type"].tolist())
    return xs, ys

class GetLabels(Transform):
    def setup(self, items, train_setup):
        if train_setup:
            self.cat = Categorize()
            self.cat.setup([x for sublist in items.items["labels"].tolist() for x in sublist])
        
    def encodes(self, x):
        labels = []
        for label in x["labels"]:
            labels.append(self.cat(label).unsqueeze(0).unsqueeze(0))
        return torch.cat(labels)

class My_Pad_Input(ItemTransform):
    def __init__(self, out_dim, from_embs=False):
        self.out_dim = out_dim
        self.from_embs = from_embs
    def encodes(self,samples, pad_fields=0, pad_first=False, backwards=False):
        "Function that collect `samples` and adds padding"
        pad_fields = L(pad_fields)
        max_len_l = pad_fields.map(lambda f: max([len(s[f]) for s in samples]))
        if backwards: pad_first = not pad_first
        def _f(field_idx, x):
            pad_value=0
            if field_idx not in pad_fields: return x
            if field_idx==1 and not self.from_embs:
                pad_value=self.out_dim
            if self.from_embs and field_idx == pad_fields[-1]:
                pad_value=self.out_dim
            idx = pad_fields.items.index(field_idx)
            sl = slice(-len(x), sys.maxsize) if pad_first else slice(0, len(x))
            pad =  x.new_zeros((max_len_l[idx]-x.shape[0], *x.shape[1:]))+pad_value
            x1 = torch.cat([pad, x] if pad_first else [x, pad])
            if backwards: x1 = x1.flip(0)
            return retain_type(x1, x)
        return [tuple(map(lambda idxx: _f(*idxx), enumerate(s))) for s in samples]
    

class GetMissingText(Transform):
    def encodes(self, x):
        nones = []
        for act in x["acts"]:
            none = tensor(True)
            text_file = act + ".npy"
            if Path(text_file).exists():
                none = tensor(False)
            nones.append(none)
        return torch.stack(nones)
    
class GetMissingImage(Transform):
    def encodes(self, x):
        nones = []
        for act in x["acts"]:
            none = tensor(True)
            img_file = act.replace("text", "img") + ".pt"
            if Path(img_file).exists():
                none = tensor(False)
            nones.append(none)
        return torch.stack(nones)

class GetImgEmbs(Transform):
    def encodes(self, x):    
        embs = []
        for act in x["acts"]:
            img_file = act.replace("text", "img") + ".pt"
            if Path(img_file).exists(): 
                img_emb = torch.load(img_file)
            else:
                img_emb = torch.zeros([4096])
            embs.append(img_emb)
        return torch.stack(embs)

    
class GetTextEmbs(Transform):
    def encodes(self, x):    
        embs = []
        for act in x["acts"]:
            text_file = act + ".npy"
            if Path(text_file).exists():
                text_emb = tensor(np.load(text_file))
            else:
                text_emb = torch.zeros([3840])
            embs.append(text_emb)
        return torch.stack(embs)
    

class GetImgAndTextEmbs(Transform):
    def encodes(self, x):    
        embs = []
        for act in x["acts"]:
            img_file = act.replace("text", "img") + ".pt"
            if Path(img_file).exists(): 
                img_emb = torch.load(img_file)
            else:
                img_emb = torch.zeros([4096])
            text_file = act + ".npy"
            if Path(text_file).exists():
                text_emb = tensor(np.load(text_file))
            else:
                text_emb = torch.zeros([3840])
            embs.append(torch.cat([img_emb, text_emb]).view(1,-1))
        return torch.cat(embs)