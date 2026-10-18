import os
try:
    import cPickle
except:
    import _pickle as cPickle
import numpy as np
from shapely.geometry import *
from tqdm import tqdm

def parse_rec_txt(filename):
    with open(filename.strip(), 'r') as f:

        gts = f.readlines()

        objects = []

        for obj in gts:

            cors = obj.strip().split(',')

            obj_struct = {}

            obj_struct['name'] = 'text'

            obj_struct['difficult'] = 0

            # bbox
            obj_struct['bbox'] = [
                int(cors[0]),
                int(cors[1]),
                int(cors[2]),
                int(cors[3])
            ]

            objects.append(obj_struct)

    return objects

def curve_parse_rec_txt(filename):
    with open(filename.strip(), 'r') as f:

        gts = f.readlines()

        objects = []

        for obj in gts:

            cors = obj.strip().split(',')

            obj_struct = {}

            obj_struct['name'] = 'text'

            obj_struct['difficult'] = 0

            obj_struct['bbox'] = [

                int(cors[0]), int(cors[1]),
                int(cors[2]), int(cors[3]),

                int(cors[4]), int(cors[5]),
                int(cors[6]), int(cors[7]),

                int(cors[8]), int(cors[9]),
                int(cors[10]), int(cors[11]),

                int(cors[12]), int(cors[13]),
                int(cors[14]), int(cors[15]),

                int(cors[16]), int(cors[17]),
                int(cors[18]), int(cors[19]),

                int(cors[20]), int(cors[21]),
                int(cors[22]), int(cors[23]),

                int(cors[24]), int(cors[25]),
                int(cors[26]), int(cors[27]),

                int(cors[28]), int(cors[29]),
                int(cors[30]), int(cors[31])
            ]

            objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):

    if use_07_metric:

        ap = 0.

        for t in np.arange(0., 1.1, 0.1):

            if np.sum(rec >= t) == 0:
                p = 0

            else:
                p = np.max(prec[rec >= t])

            ap = ap + p / 11.

    else:

        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):

            mpre[i - 1] = np.maximum(
                mpre[i - 1],
                mpre[i]
            )

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum(
            (mrec[i + 1] - mrec[i]) *
            mpre[i + 1]
        )

    return ap

def voc_eval_polygon(
        detpath,
        annopath,
        imagesetfile,
        classname,
        ovthresh=0.5,
        use_07_metric=False):

    cachefile = 'dataset/ctw1500_eval/annots.pkl'

    with open(imagesetfile, 'r') as f, \
            open(annopath, 'r') as fa:

        lines = f.readlines()
        anno_lines = fa.readlines()

    imagenames = [x.strip() for x in lines]
    anno_names = [y.strip() for y in anno_lines]

    assert (
        len(imagenames) == len(anno_names)
    ), 'each image should correspond to one label file'

    if not os.path.isfile(cachefile):

        recs = {}

        for i, imagename in enumerate(imagenames):

            print(anno_names[i].strip())

            recs[imagename] = curve_parse_rec_txt(
                anno_names[i]
            )

            if i % 100 == 0:

                print(
                    'Reading annotation for {:d}/{:d}'.format(
                        i + 1,
                        len(imagenames)
                    )
                )

        print('Saving cached annotations to {:s}'.format(cachefile))

        with open(cachefile, 'wb') as f:

            cPickle.dump(recs, f)

    else:

        with open(cachefile, 'rb') as f:

            recs = cPickle.load(f)
    class_recs = {}

    npos = 0

    num_gt = {}

    for ix, imagename in enumerate(imagenames):

        R = [
            obj for obj in recs[imagename]
            if obj['name'] == classname
        ]

        if not R:
            continue

        bbox = np.array([x['bbox'] for x in R])

        # difficult
        difficult = np.array(
            [x['difficult'] for x in R]
        ).astype(np.bool)

        det = [False] * len(R)

        npos = npos + sum(~difficult)

        num_gt[str(ix)] = sum(~difficult)

        class_recs[str(ix)] = {
            'bbox': bbox,
            'difficult': difficult,
            'det': det
        }

    detfile = detpath.format(classname)

    with open(detfile, 'r') as f:

        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]

    image_ids = [x[0] for x in splitlines]

    # confidence
    confidence = np.array(
        [float(x[1]) for x in splitlines]
    )

    BB = np.array([
        [float(z) for z in x[2:]]
        for x in splitlines
    ])

    sorted_ind = np.argsort(-confidence)

    sorted_scores = np.sort(-confidence)

    nd = len(image_ids)

    # true positive
    tp = np.zeros(nd)

    # false positive
    fp = np.zeros(nd)

    for d in tqdm(range(nd)):

        R = class_recs[image_ids[d]]

        bb = BB[d]

        det_bbox = bb[:]

        pts = [
            (det_bbox[j], det_bbox[j + 1])
            for j in range(0, len(bb), 2)
        ]

        try:
            pdet = Polygon(pts)

        except Exception as e:
            print(e)
            continue

        if not pdet.is_valid:
            continue

        ovmax = -np.inf

        BBGT = R['bbox'].astype(float)

        gt_bbox = BBGT[:, :4]

        info_bbox_gt = BBGT[:, 4:32]

        overlaps = np.zeros(BBGT.shape[0])

        for iix in range(BBGT.shape[0]):

            pts = [

                (
                    int(gt_bbox[iix, 0]) + info_bbox_gt[iix, j],

                    int(gt_bbox[iix, 1]) + info_bbox_gt[iix, j + 1]

                )

                for j in range(0, 28, 2)

            ]

            pgt = Polygon(pts)

            if not pgt.is_valid:
                print('GT polygon has intersecting sides.')
                continue

            # intersection
            try:
                sec = pdet.intersection(pgt)

            except Exception as e:
                print('intersect invalid', e)
                continue

            try:
                assert sec.is_valid

            except Exception as e:
                print(e)
                continue

            inters = sec.area

            uni = pgt.area + pdet.area - inters

            if uni <= 0.00001:
                uni = 0.00001

            # IoU
            overlaps[iix] = inters * 1.0 / uni

        ovmax = np.max(overlaps)

        jmax = np.argmax(overlaps)

        if ovmax > ovthresh:

            if not R['difficult'][jmax]:

                if not R['det'][jmax]:

                    tp[d] = 1.

                    R['det'][jmax] = 1

                else:

                    fp[d] = 1.

        else:

            fp[d] = 1.

    fpp = fp
    tpp = tp

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    # recall
    rec = tp / float(npos)

    # precision
    prec = tp / np.maximum(
        tp + fp,
        np.finfo(np.float64).eps
    )

    # AP
    ap = voc_ap(
        rec,
        prec,
        use_07_metric
    )

    return rec, prec, ap, fpp, tpp, image_ids, num_gt