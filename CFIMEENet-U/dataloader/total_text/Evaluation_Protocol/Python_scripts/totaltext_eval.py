import warnings
warnings.filterwarnings("ignore")
import numpy as np
from os import listdir
from scipy import io
from base_function import iod
from base_function import area_of_intersection
from base_function import area
import cv2
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('exp_name',type=str,help='Model output directory')

parser.add_argument(
    '--tr',
    type=float,
    default=0.7,
    help='Recall threshold'
)

parser.add_argument(
    '--tp',
    type=float,
    default=0.6,
    help='Precision threshold'
)

args = parser.parse_args()

"""
y0,x0,...,yn,xn
"""

input_dir = 'data/output/{}'.format(args.exp_name)

gt_dir = 'data/total-text-mat/labels/Test'

allInputs = listdir(input_dir)


def input_reading_mod(input_dir, input):
    """
    """

    with open('%s/%s' % (input_dir, input), 'r') as input_fid:
        pred = input_fid.readlines()

    det = [x.strip('\n') for x in pred]

    return det


def gt_reading_mod(gt_dir, gt_id):
    """
    """

    gt_id = gt_id.split('.')[0]

    gt = io.loadmat(
        '%s/poly_gt_%s.mat' % (gt_dir, gt_id)
    )

    gt = gt['polygt']

    return gt


def detection_filtering(detections, groundtruths, threshold=0.5):
    """
    """

    for gt_id, gt in enumerate(groundtruths):

        if (gt[5] == '#') and (gt[1].shape[1] > 1):

            gt_x = list(map(int, np.squeeze(gt[1])))
            gt_y = list(map(int, np.squeeze(gt[3])))

            for det_id, detection in enumerate(detections):

                detection = detection.split(',')

                detection = list(map(int, detection[0:]))

                det_y = detection[1::2]
                det_x = detection[0::2]

                det_gt_iou = iod(
                    det_x,
                    det_y,
                    gt_x,
                    gt_y
                )

                if det_gt_iou > threshold:
                    detections[det_id] = []

            detections[:] = [
                item for item in detections
                if item != []
            ]

    return detections


def sigma_calculation(det_x, det_y, gt_x, gt_y):
    """
    sigma = intersection / annotation area
    """

    return np.round(
        (
            area_of_intersection(
                det_x,
                det_y,
                gt_x,
                gt_y
            ) / area(gt_x, gt_y)
        ),
        2
    )


def tau_calculation(det_x, det_y, gt_x, gt_y):
    """
    tau = intersection / prediction area
    """

    return np.round(
        (
            area_of_intersection(
                det_x,
                det_y,
                gt_x,
                gt_y
            ) / area(det_x, det_y)
        ),
        2
    )

global_tp = 0
global_fp = 0
global_fn = 0

global_sigma = []
global_tau = []

tr = args.tr
tp = args.tp

fsc_k = 0.8

k = 2

global_size_table = []

gt_small = 0
gt_medium = 0
gt_large = 0

for i, input_id in enumerate(tqdm(allInputs)):

    if (
        input_id != '.DS_Store'
        and input_id != 'Pascal_result.txt'
        and input_id != 'Pascal_result_curved.txt'
        and input_id != 'Pascal_result_non_curved.txt'
        and input_id != 'Deteval_result.txt'
        and input_id != 'Deteval_result_curved.txt'
        and input_id != 'Deteval_result_non_curved.txt'
    ):

        detections = input_reading_mod(
            input_dir,
            input_id
        )

        groundtruths = gt_reading_mod(
            gt_dir,
            input_id
        )

        detections = detection_filtering(
            detections,
            groundtruths
        )

        dc_id = np.where(groundtruths[:, 5] == '#')

        groundtruths = np.delete(
            groundtruths,
            (dc_id),
            (0)
        )

        # sigma table
        local_sigma_table = np.zeros(
            (groundtruths.shape[0], len(detections))
        )

        # tau table
        local_tau_table = np.zeros(
            (groundtruths.shape[0], len(detections))
        )

        local_size_table = np.zeros(len(detections))

        flag = True

        for gt_id, gt in enumerate(groundtruths):

            gt_x = list(map(int, np.squeeze(gt[1])))
            gt_y = list(map(int, np.squeeze(gt[3])))

            pts = np.array(
                np.stack([gt_x, gt_y])
            ).T

            pts_area = cv2.contourArea(pts)

            if pts_area < 1000:
                gt_small += 1

            elif pts_area < 10000:
                gt_medium += 1

            else:
                gt_large += 1

            if len(detections) > 0:

                for det_id, detection in enumerate(detections):

                    detection = detection.split(',')

                    detection = list(map(int, detection))

                    det_y = detection[1::2]
                    det_x = detection[0::2]

                    gt_x = list(map(int, np.squeeze(gt[1])))
                    gt_y = list(map(int, np.squeeze(gt[3])))

                    if flag:

                        pts = np.array(detection).reshape(-1, 2)

                        pts_area = cv2.contourArea(pts)

                        if pts_area < 1000:
                            local_size_table[det_id] = 1

                        elif pts_area < 10000:
                            local_size_table[det_id] = 2

                        else:
                            local_size_table[det_id] = 3

                    local_sigma_table[gt_id, det_id] = sigma_calculation(
                        det_x,
                        det_y,
                        gt_x,
                        gt_y
                    )

                    local_tau_table[gt_id, det_id] = tau_calculation(
                        det_x,
                        det_y,
                        gt_x,
                        gt_y
                    )

                if flag:
                    flag = False

        global_sigma.append(local_sigma_table)
        global_tau.append(local_tau_table)
        global_size_table.append(local_size_table)

global_accumulative_recall = 0
global_accumulative_precision = 0

total_num_gt = 0

total_num_det = 0

def one_to_one(
        local_sigma_table,
        local_tau_table,
        local_accumulative_recall,
        local_accumulative_precision,
        global_accumulative_recall,
        global_accumulative_precision,
        gt_flag,
        det_flag):
    """
    """

    for gt_id in range(num_gt):

        qualified_sigma_candidates = np.where(
            local_sigma_table[gt_id, :] > tr
        )

        num_qualified_sigma_candidates = \
            qualified_sigma_candidates[0].shape[0]

        qualified_tau_candidates = np.where(
            local_tau_table[gt_id, :] > tp
        )

        num_qualified_tau_candidates = \
            qualified_tau_candidates[0].shape[0]

        if (
            num_qualified_sigma_candidates == 1
            and num_qualified_tau_candidates == 1
        ):

            global_accumulative_recall += 1.0
            global_accumulative_precision += 1.0

            local_accumulative_recall += 1.0
            local_accumulative_precision += 1.0

            gt_flag[0, gt_id] = 1

            matched_det_id = np.where(
                local_sigma_table[gt_id, :] > tr
            )

            det_flag[0, matched_det_id] = 1

    return (
        local_accumulative_recall,
        local_accumulative_precision,
        global_accumulative_recall,
        global_accumulative_precision,
        gt_flag,
        det_flag
    )


def one_to_many(
        local_sigma_table,
        local_tau_table,
        local_accumulative_recall,
        local_accumulative_precision,
        global_accumulative_recall,
        global_accumulative_precision,
        gt_flag,
        det_flag):
    """
    """

    for gt_id in range(num_gt):

        if gt_flag[0, gt_id] > 0:
            continue

        non_zero_in_sigma = np.where(
            local_sigma_table[gt_id, :] > 0
        )

        num_non_zero_in_sigma = \
            non_zero_in_sigma[0].shape[0]

        if num_non_zero_in_sigma >= k:

            qualified_tau_candidates = np.where(
                (local_tau_table[gt_id, :] >= tp)
                & (det_flag[0, :] == 0)
            )

            num_qualified_tau_candidates = \
                qualified_tau_candidates[0].shape[0]

            if num_qualified_tau_candidates == 1:

                if (
                    (
                        local_tau_table[
                            gt_id,
                            qualified_tau_candidates
                        ] >= tp
                    )
                    and
                    (
                        local_sigma_table[
                            gt_id,
                            qualified_tau_candidates
                        ] >= tr
                    )
                ):

                    global_accumulative_recall += 1.0
                    global_accumulative_precision += 1.0

                    local_accumulative_recall += 1.0
                    local_accumulative_precision += 1.0

                    gt_flag[0, gt_id] = 1

                    det_flag[0, qualified_tau_candidates] = 1

            elif (
                np.sum(
                    local_sigma_table[
                        gt_id,
                        qualified_tau_candidates
                    ]
                ) >= tr
            ):

                gt_flag[0, gt_id] = 1

                det_flag[0, qualified_tau_candidates] = 1

                global_accumulative_recall += fsc_k

                global_accumulative_precision += \
                    num_qualified_tau_candidates * fsc_k

                local_accumulative_recall += fsc_k

                local_accumulative_precision += \
                    num_qualified_tau_candidates * fsc_k

    return (
        local_accumulative_recall,
        local_accumulative_precision,
        global_accumulative_recall,
        global_accumulative_precision,
        gt_flag,
        det_flag
    )


def many_to_many(
        local_sigma_table,
        local_tau_table,
        local_accumulative_recall,
        local_accumulative_precision,
        global_accumulative_recall,
        global_accumulative_precision,
        gt_flag,
        det_flag):
    """
    """

    for det_id in range(num_det):

        if det_flag[0, det_id] > 0:
            continue

        non_zero_in_tau = np.where(
            local_tau_table[:, det_id] > 0
        )

        num_non_zero_in_tau = \
            non_zero_in_tau[0].shape[0]

        if num_non_zero_in_tau >= k:

            qualified_sigma_candidates = np.where(
                (local_sigma_table[:, det_id] >= tp)
                & (gt_flag[0, :] == 0)
            )

            num_qualified_sigma_candidates = \
                qualified_sigma_candidates[0].shape[0]

            if num_qualified_sigma_candidates == 1:

                if (
                    (
                        local_tau_table[
                            qualified_sigma_candidates,
                            det_id
                        ] >= tp
                    )
                    and
                    (
                        local_sigma_table[
                            qualified_sigma_candidates,
                            det_id
                        ] >= tr
                    )
                ):

                    global_accumulative_recall += 1.0
                    global_accumulative_precision += 1.0

                    local_accumulative_recall += 1.0
                    local_accumulative_precision += 1.0

                    gt_flag[0, qualified_sigma_candidates] = 1

                    det_flag[0, det_id] = 1

            elif (
                np.sum(
                    local_tau_table[
                        qualified_sigma_candidates,
                        det_id
                    ]
                ) >= tp
            ):

                det_flag[0, det_id] = 1

                gt_flag[0, qualified_sigma_candidates] = 1

                global_accumulative_recall += \
                    num_qualified_sigma_candidates * fsc_k

                global_accumulative_precision += fsc_k

                local_accumulative_recall += \
                    num_qualified_sigma_candidates * fsc_k

                local_accumulative_precision += fsc_k

    return (
        local_accumulative_recall,
        local_accumulative_precision,
        global_accumulative_recall,
        global_accumulative_precision,
        gt_flag,
        det_flag
    )

small_match = 0
middle_match = 0
large_match = 0
sumofmatch = 0

eval_result_dir = "data/output/Analysis/output_eval"

fid_path = '{}/Eval_TotalText_{}_{}.txt'.format(
    eval_result_dir,
    args.tr,
    args.tp
)

fid = open(fid_path, 'w')

for idx in range(len(global_sigma)):

    local_sigma_table = global_sigma[idx]
    local_tau_table = global_tau[idx]

    num_gt = local_sigma_table.shape[0]
    num_det = local_sigma_table.shape[1]

    total_num_gt += num_gt
    total_num_det += num_det

    local_accumulative_recall = 0
    local_accumulative_precision = 0

    gt_flag = np.zeros((1, num_gt))
    det_flag = np.zeros((1, num_det))

    local_accumulative_recall, \
    local_accumulative_precision, \
    global_accumulative_recall, \
    global_accumulative_precision, \
    gt_flag, \
    det_flag = one_to_one(
        local_sigma_table,
        local_tau_table,
        local_accumulative_recall,
        local_accumulative_precision,
        global_accumulative_recall,
        global_accumulative_precision,
        gt_flag,
        det_flag
    )

    local_accumulative_recall, \
    local_accumulative_precision, \
    global_accumulative_recall, \
    global_accumulative_precision, \
    gt_flag, \
    det_flag = one_to_many(
        local_sigma_table,
        local_tau_table,
        local_accumulative_recall,
        local_accumulative_precision,
        global_accumulative_recall,
        global_accumulative_precision,
        gt_flag,
        det_flag
    )

    local_accumulative_recall, \
    local_accumulative_precision, \
    global_accumulative_recall, \
    global_accumulative_precision, \
    gt_flag, \
    det_flag = many_to_many(
        local_sigma_table,
        local_tau_table,
        local_accumulative_recall,
        local_accumulative_precision,
        global_accumulative_recall,
        global_accumulative_precision,
        gt_flag,
        det_flag
    )

    # precision
    try:
        local_precision = \
            local_accumulative_precision / num_det

    except ZeroDivisionError:
        local_precision = 0

    # recall
    try:
        local_recall = \
            local_accumulative_recall / num_gt

    except ZeroDivisionError:
        local_recall = 0

    local_size_table = global_size_table[idx]

    match_table = local_size_table[
        det_flag[0] == 1
    ]

    for i in match_table:

        sumofmatch += 1

        if i == 1:
            small_match += 1

        elif i == 2:
            middle_match += 1

        elif i == 3:
            large_match += 1

    str_write = (
        '%s :: Precision=%.4f - Recall=%.4f\n'
        % (
            allInputs[idx],
            local_precision,
            local_recall
        )
    )

    fid.write(str_write)

fid.close()

try:
    recall = global_accumulative_recall / total_num_gt

except ZeroDivisionError:
    recall = 0

try:
    precision = global_accumulative_precision / total_num_det

except ZeroDivisionError:
    precision = 0

# F-score
try:
    f_score = \
        2 * precision * recall / (precision + recall)

except ZeroDivisionError:
    f_score = 0

fid = open(fid_path, 'a')

str_write = (
    'ALL :: Precision=%.4f - Recall=%.4f - Fscore=%.4f'
    % (
        precision,
        recall,
        f_score
    )
)

fid.write(str_write)

fid.close()

print('Input: {}'.format(input_dir))

print(
    'Config: tr: {} - tp: {}'.format(
        tr,
        tp
    )
)

print(str_write)

print(
    "small/middle/large/sum : "
    "{}/{}/{}/{}".format(
        small_match,
        middle_match,
        large_match,
        sumofmatch
    )
)

print(
    "annotation : small/middle/large : "
    "{}/{}/{}".format(
        gt_small,
        gt_medium,
        gt_large
    )
)

print(
    "annotation% : small/middle/large : "
    "{}/{}/{}".format(
        small_match / gt_small,
        middle_match / gt_medium,
        large_match / gt_large
    )
)

print('Done.\n')