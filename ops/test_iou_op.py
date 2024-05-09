import torch
from iou3d_op import nms_gpu, boxes_overlap_bev_gpu, boxes_iou_bev_gpu


def test_overlap_bev():
    boxes = torch.Tensor([
        [0.0, 0.0, 0.5, 0.5, 0.0], 
        [0.3, 0.0, 0.5, 0.5, 0.0],
        [0.03, 0.0, 0.5, 0.5, 0.0],
        [-0.5, 0.0, 0.5, 0.5, 0.0],
        [0.2, 0.2, 0.5, 0.5, 0.0],
        [0.25, 0.25, 0.5, 0.5, 0.0]
        ])
    
    ans_overlap = boxes.new_zeros(
        torch.Size((boxes.shape[0], boxes.shape[0])))

    boxes_overlap_bev_gpu(boxes, boxes, ans_overlap)
    print(ans_overlap)


def test_iou_bev():
    boxes = torch.Tensor([
        [0.0, 0.0, 0.5, 0.5, 0.0], 
        [0.3, 0.0, 0.5, 0.5, 0.0],
        [0.03, 0.0, 0.5, 0.5, 0.0],
        [-0.5, 0.0, 0.5, 0.5, 0.0],
        [0.2, 0.2, 0.5, 0.5, 0.0],
        [0.25, 0.25, 0.5, 0.5, 0.0]
        ])
    
    ans_overlap = boxes.new_zeros(
        torch.Size((boxes.shape[0], boxes.shape[0])))

    boxes_iou_bev_gpu(boxes, boxes, ans_overlap)
    print(ans_overlap)


def test_nms_gpu():
    boxes = torch.Tensor([
        [0.0, 0.0, 0.5, 0.5, 0.0], 
        [0.3, 0.0, 0.5, 0.5, 0.0],
        [0.03, 0.0, 0.5, 0.5, 0.0],
        [-0.5, 0.0, 0.5, 0.5, 0.0],
        [0.2, 0.2, 0.5, 0.5, 0.0],
        [0.25, 0.25, 0.5, 0.5, 0.0]
        ])
    keep = torch.zeros_like(boxes)
    ret = nms_gpu(boxes, keep, 0.7, 1)
    print(ret)
    print(keep)
    pass


if __name__ == "__main__":
    test_nms_gpu()
    test_iou_bev()
    test_overlap_bev()