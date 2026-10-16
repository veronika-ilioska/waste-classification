"""Train Mask R-CNN on ZeroWaste's official splits; select by validation mask AP."""
from __future__ import annotations

import argparse
import copy
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor

CLASSES = ['background', 'rigid_plastic', 'cardboard', 'metal', 'soft_plastic']


def extract(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f'Unsafe archive path: {entry.filename}')
        print(f'Extracting {archive} to {destination}', flush=True)
        z.extractall(destination)


class ZeroWaste(Dataset):
    def __init__(self, root, split, augment=False):
        self.folder = root / split
        self.coco = COCO(str(self.folder / 'labels.json'))
        categories = {c['id']: c['name'] for c in self.coco.dataset['categories']}
        if categories != dict(enumerate(CLASSES[1:], 1)):
            raise ValueError(f'Unexpected categories: {categories}')
        # COCO matching uses zero as the unmatched sentinel: IDs must be positive.
        for index, ann in enumerate(self.coco.dataset['annotations'], 1):
            ann['id'] = index
        self.coco.createIndex()
        self.ids = sorted(self.coco.imgs)
        self.augment = augment
        missing = [i for i in self.ids if not self.image_path(i).is_file()]
        if missing:
            raise FileNotFoundError(f'{split}: missing {len(missing)} images')

    def image_path(self, image_id):
        return self.folder / 'data' / self.coco.imgs[image_id]['file_name']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        with Image.open(self.image_path(image_id)) as source:
            image = pil_to_tensor(source.convert('RGB')).float() / 255
        h, w = image.shape[-2:]
        boxes, labels, masks, crowds = [], [], [], []
        for ann in self.coco.imgToAnns.get(image_id, []):
            x, y, bw, bh = ann['bbox']
            box = [max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)]
            mask = self.coco.annToMask(ann)
            if box[2] <= box[0] or box[3] <= box[1] or not mask.any():
                raise ValueError(f'Invalid annotation {ann["id"]} in {self.folder}')
            boxes.append(box)
            labels.append(ann['category_id'])
            masks.append(mask)
            crowds.append(ann.get('iscrowd', 0))
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        masks = torch.from_numpy(np.stack(masks) if masks else np.zeros((0, h, w), dtype=np.uint8))
        if self.augment and random.random() < 0.5:
            image = image.flip(-1)
            masks = masks.flip(-1)
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return image, {'boxes': boxes, 'labels': torch.tensor(labels, dtype=torch.int64),
                       'masks': masks, 'image_id': torch.tensor(image_id),
                       'area': masks.flatten(1).sum(1).float(),
                       'iscrowd': torch.tensor(crowds, dtype=torch.int64)}


def collate(batch):
    return tuple(zip(*batch))


def build_model(pretrained=True, min_size=800):
    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
        weights_backbone=None, min_size=min_size,
        max_size=1333 if min_size == 800 else min_size * 2,
        box_score_thresh=0.001)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, len(CLASSES))
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, len(CLASSES))
    return model


@torch.inference_mode()
def evaluate(model, dataset, device, output):
    model.eval()
    results = []
    for index in range(len(dataset)):
        image, target = dataset[index]
        pred = model([image.to(device)])[0]
        for box, label, score, mask in zip(pred['boxes'].cpu(), pred['labels'].cpu(),
                                         pred['scores'].cpu(), pred['masks'].cpu()):
            binary = (mask[0].numpy() >= 0.5).astype(np.uint8)
            if not binary.any():
                continue
            rle = mask_utils.encode(np.asfortranarray(binary))
            rle['counts'] = rle['counts'].decode('ascii')
            x1, y1, x2, y2 = box.tolist()
            results.append(dict(image_id=int(target['image_id']), category_id=int(label),
                                score=float(score), bbox=[x1, y1, x2-x1, y2-y1], segmentation=rle))
        if (index + 1) % 50 == 0:
            print(f'Evaluated {index + 1}/{len(dataset)} images', flush=True)
    gt = dataset.coco
    gt.dataset.setdefault('info', {})
    if results:
        detected = {kind: gt.loadRes([{k: v for k, v in r.items() if kind != 'segm' or k != 'bbox'}
                                    for r in results]) for kind in ('bbox', 'segm')}
    else:
        empty = COCO()
        empty.dataset = copy.deepcopy(gt.dataset)
        empty.dataset['annotations'] = []
        empty.createIndex()
        detected = {'bbox': empty, 'segm': empty}
    metrics = {}
    for kind in ('bbox', 'segm'):
        evaluator = COCOeval(gt, detected[kind], kind)
        evaluator.params.imgIds = dataset.ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        metrics[kind] = dict(zip(['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
                                 'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large'],
                                map(float, evaluator.stats)))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (output / 'predictions.json').write_text(json.dumps(results))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('datasets/ZeroWaste/splits_final_deblurred'))
    parser.add_argument('--archive', type=Path, help='Extract downloaded ZIP into --data parent first')
    parser.add_argument('--output', type=Path, default=Path('artifacts/zerowaste/maskrcnn'))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0025)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--check-only', action='store_true', help='One training step and validation inference, no pretrained download')
    parser.add_argument('--evaluate', type=Path, help='Evaluate a saved checkpoint on the official test split')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('Epochs and batch size must be positive')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if args.archive:
        extract(args.archive, args.data.parent)
    datasets = {split: ZeroWaste(args.data, split, augment=split == 'train')
                for split in ('train', 'val', 'test')}
    print({split: len(ds) for split, ds in datasets.items()}, flush=True)
    device = torch.device(args.device)
    if device.type == 'cpu' and not args.check_only and not args.evaluate:
        parser.error('Full training needs a GPU for practical runtime. Run this script in a GPU allocation on cuda4 with CUDA-enabled PyTorch; --check-only works on CPU.')
    if not args.check_only and not args.evaluate and (args.output / 'best_model.pth').exists():
        parser.error('Output already contains a checkpoint. Choose a new --output directory.')
    model = build_model(not args.check_only and not args.evaluate,
                        min_size=128 if args.check_only else 800).to(device)
    if args.evaluate:
        model.load_state_dict(torch.load(args.evaluate, map_location=device, weights_only=True))
        evaluate(model, datasets['test'], device, args.output / 'test')
        return
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr, momentum=0.9, weight_decay=0.0005)
    if args.check_only:
        model.train()
        image, target = datasets['train'][0]
        losses = model([image.to(device)], [{k: v.to(device) for k, v in target.items()}])
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite smoke loss')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            pred = model([datasets['val'][0][0].to(device)])[0]
        print('PASS: real mask targets, five training losses, backward/optimizer step, validation inference', flush=True)
        print({k: float(v.detach()) for k, v in losses.items()})
        print({k: list(v.shape) for k, v in pred.items()})
        return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'config.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                                       for k, v in vars(args).items()}, indent=2))
    (args.output / 'labels.json').write_text(json.dumps(CLASSES, indent=2))
    loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, collate_fn=collate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    history, best, stale = [], -float('inf'), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, (images, targets) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            losses = model([im.to(device) for im in images],
                           [{k: v.to(device) for k, v in t.items()} for t in targets])
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch}, step {step}')
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            if step % 50 == 0:
                print(f'Epoch {epoch}, batch {step}/{len(loader)}, loss {total/step:.4f}', flush=True)
        metrics = evaluate(model, datasets['val'], device, args.output / 'validation' / f'epoch_{epoch:03d}')
        ap = metrics['segm']['AP']
        history.append(dict(epoch=epoch, loss=total/len(loader), validation_mask_AP=ap))
        (args.output / 'history.json').write_text(json.dumps(history, indent=2))
        if ap > best:
            best, stale = ap, 0
            torch.save(model.state_dict(), args.output / 'best_model.pth')
            (args.output / 'best_epoch.json').write_text(json.dumps(history[-1], indent=2))
        else:
            stale += 1
        scheduler.step()
        print(history[-1], flush=True)
        if args.patience > 0 and stale >= args.patience:
            break
    model.load_state_dict(torch.load(args.output / 'best_model.pth', map_location=device, weights_only=True))
    evaluate(model, datasets['test'], device, args.output / 'test')
    print(f'Completed. Best validation checkpoint and test metrics: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
