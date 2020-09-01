import mmcv
import numpy as np
from tqdm import tqdm
from mmdet.ops.nms import nms_wrapper
# from mmdet.core.bbox import bbox_overlaps
from mmdet.ops.bbox import cython_bbox
import json
import datetime
from mmdet import datasets


def get_names(predicted_file):
    names = []
    for predict in predicted_file:
        names.append(predict['name'])
    return list(set(names))

def box_voting(top_dets, all_dets, thresh, scoring_method='ID', beta=1.0):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """
    # top_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    # all_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    top_dets_out = top_dets.copy()
    top_boxes = top_dets[:, :4]
    all_boxes = all_dets[:, :4]
    all_scores = all_dets[:, 4]
    top_to_all_overlaps = bbox_overlaps(top_boxes, all_boxes)
    for k in range(top_dets_out.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
        if scoring_method == 'ID':
            # Identity, nothing to do
            pass
        elif scoring_method == 'TEMP_AVG':
            # Average probabilities (considered as P(detected class) vs.
            # P(not the detected class)) after smoothing with a temperature
            # hyperparameter.
            P = np.vstack((ws, 1.0 - ws))
            P_max = np.max(P, axis=0)
            X = np.log(P / P_max)
            X_exp = np.exp(X / beta)
            P_temp = X_exp / np.sum(X_exp, axis=0)
            P_avg = P_temp[0].mean()
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'AVG':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.mean()
        elif scoring_method == 'IOU_AVG':
            P = ws
            ws = top_to_all_overlaps[k, inds_to_vote]
            P_avg = np.average(P, weights=ws)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'GENERALIZED_AVG':
            P_avg = np.mean(ws**beta)**(1.0 / beta)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'QUASI_SUM':
            top_dets_out[k, 4] = ws.sum() / float(len(ws))**beta
        else:
            raise NotImplementedError(
                'Unknown scoring method {}'.format(scoring_method)
            )

    return top_dets_out

if __name__ == "__main__":
    bbox_overlaps = cython_bbox.bbox_overlaps
    nms_type = 'nms'
    nms_op = getattr(nms_wrapper, nms_type)
    ROOT_DIR = "./"
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    model1 = mmcv.load(ROOT_DIR + "m1.pkl")
    model2 = mmcv.load(ROOT_DIR + "m2.pkl")
    model3 = mmcv.load(ROOT_DIR + "m3.pkl")
    cfg = mmcv.Config.fromfile(r'custom_configs/eccv_coco_detectors_resnest152_12x_m3.py')
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
#     names1 = get_names(model1)
#     names2 = get_names(model2)
    final_names = [test_dataset.img_infos[i]['filename'] for i in range(len(test_dataset))]
    print(len(model1), len(model3))
    print(len(model1[0]), len(model3[0]))

    result = {name: [[] for i in range(len(CLASSES))] for name in final_names}
   
    for i, predict in tqdm(enumerate(model1)):
#         name = predict['name']
        name = test_dataset.img_infos[i]['filename']
        if name not in final_names:
            continue
#         cls = predict['category']
#         score = predict['score']
#         bbox = predict['bbox']
#         bbox = bbox + [score]
        for j in range(len(CLASSES)):
            for k in range(model1[i][j].shape[0]):
                result[name][j].append(model1[i][j][k])
                
#         result[name] = model1[i]

    for i, predict in tqdm(enumerate(model2)):
#         name = predict['name']
        name = test_dataset.img_infos[i]['filename']
        if name not in final_names:
            continue
#         cls = predict['category']
#         score = predict['score']
#         bbox = predict['bbox']
#         bbox = bbox + [score]
        for j in range(len(CLASSES)):
            for k in range(model2[i][j].shape[0]):
                result[name][j].append(model2[i][j][k])
                
#             result[name][j].extend( model2[i][j])
  
    for i, predict in tqdm(enumerate(model3)):
#         name = predict['name']
        name = test_dataset.img_infos[i]['filename']
        if name not in final_names:
            continue
        for j in range(len(CLASSES)):
            for k in range(model3[i][j].shape[0]):
                result[name][j].append(model3[i][j][k])

        
    submit = []
    temp = []
    for name in tqdm(final_names):
        for i in range(0, len(CLASSES)):
            det = np.array(result[name][i])
            
            if det.shape[0] == 0:
                continue
            cls_dets, _ = nms_op(det, iou_thr=0.5)

            cls_dets = box_voting(np.array(cls_dets, dtype=np.float32), np.array(det, np.float32), thresh=0.7,
                                  scoring_method='ID', beta=1.0)
            for bbox in cls_dets:
                res_line = {'name': name, 'category': int(i), 'bbox': [round(float(x), 2) for x in bbox[:4]],
                            'score': float(bbox[4])}
                submit.append(res_line)
            for bbox in det:
                res_line = {'name': name, 'category': int(i), 'bbox': [round(float(x), 2) for x in bbox[:4]],
                            'score': float(bbox[4])}
                temp.append(res_line)
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    out = "ensemble_result_{}.json".format(nowTime)
    with open(out, 'w') as fp:
        json.dump(submit, fp, indent=4, separators=(',', ': '))
    with open('ddd.json', 'w') as fp:
        json.dump(temp, fp, indent=4, separators=(',', ': '))
    print('over!')