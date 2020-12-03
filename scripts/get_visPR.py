# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
sys.path.append("/mnt/disk3/chen/workshop/thesis/CrowdPose/crowdpose-api/PythonAPI")
from crowdposetools.coco import COCO
from crowdposetools.cocoeval import COCOeval


# %%
gt_file = "/mnt/disk3/chen/workshop/thesis/human-pose-estimation.pytorch/data/crowdpose/annotations/crowdpose_test_vis.json"
res_file = "/mnt/disk3/chen/workshop/thesis/human-pose-estimation.pytorch/output/crowdpose/pose_resnet_del_unvis_1x1_50/del_unvis_1x1_256x192_d256x3_adam_lr1e-3/results/keypoints_test_results.json"

coco = COCO(gt_file)
coco_dt = coco.loadRes(res_file)
coco_eval = COCOeval(coco, coco_dt, 'keypoints')
coco_eval.params.useSegm = None


# %%
coco_eval.evaluate()


# %%
coco_eval.accumulate()


# %%
coco_eval.summarize()


# %%



