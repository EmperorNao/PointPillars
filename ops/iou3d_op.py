import torch
from math import cos, sin, atan2


EPS = 1e-8

class Point:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def set(self, _x, _y):
        self.x = _x
        self.y = _y

    def __add__(self, b):
        return Point(self.x + b.x, self.y + b.y)

    def __sub__(self, b):
        return Point(self.x - b.x, self.y - b.y)


def rotate_around_center(center, angle_cos, angle_sin, p):
    new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * angle_sin + center.x
    new_y = -(p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y
    p.set(new_x, new_y)


def check_rect_cross(p1, p2, q1, q2):
    ret = min(p1.x, p2.x) <= max(q1.x, q2.x) and \
        min(q1.x, q2.x) <= max(p1.x, p2.x) and \
        min(p1.y, p2.y) <= max(q1.y, q2.y) and \
        min(q1.y, q2.y) <= max(p1.y, p2.y)
    return ret




def cross(p1, p2, p0=None):
    if p0 is None:
        return p1.x * p2.y - p1.y * p2.x
    else:
        return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)


def intersection(p1, p0, q1, q0, ans):
    if (check_rect_cross(p0, p1, q0, q1) == 0):
        return 0

    s1 = cross(q0, p1, p0)
    s2 = cross(p1, q1, p0)
    s3 = cross(p0, q1, q0)
    s4 = cross(q1, p1, q0)

    if (not(s1 * s2 > 0 and s3 * s4 > 0)):
        return 0

    s5 = cross(q1, p1, p0)
    if (abs(s5 - s1) > EPS):
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1)
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1)

    else:
        a0 = p0.y - p1.y
        b0 = p1.x - p0.x
        c0 = p0.x * p1.y - p1.x * p0.y
        a1 = q0.y - q1.y
        b1 = q1.x - q0.x
        c1 = q0.x * q1.y - q1.x * q0.y
        D = a0 * b1 - a1 * b0

        ans.x = (b0 * c1 - b1 * c0) / D
        ans.y = (a1 * c0 - a0 * c1) / D

    return 1


def check_in_box2d(box, p):
    MARGIN = 1e-5

    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    angle_cos = cos(-box[4])
    angle_sin = sin(-box[4])
    rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * angle_sin + center_x
    rot_y = -(p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos + center_y

    return (rot_x > box[0] - MARGIN and rot_x < box[2] + MARGIN and
          rot_y > box[1] - MARGIN and rot_y < box[3] + MARGIN)


def point_cmp(a, b, center):
    return atan2(a.y - center.y, a.x - center.x) > \
        atan2(b.y - center.y, b.x - center.x)


def box_overlap(box_a, box_b):
    a_x1 = box_a[0]
    a_y1 = box_a[1]
    a_x2 = box_a[2]
    a_y2 = box_a[3]
    a_angle = box_a[4]

    b_x1 = box_b[0]
    b_y1 = box_b[1]
    b_x2 = box_b[2]
    b_y2 = box_b[3]
    b_angle = box_b[4]

    center_a = Point((a_x1 + a_x2) / 2, (a_y1 + a_y2) / 2)
    center_b = Point((b_x1 + b_x2) / 2, (b_y1 + b_y2) / 2)

    box_a_corners = [Point() for i in range(5)]
    box_a_corners[0].set(a_x1, a_y1)
    box_a_corners[1].set(a_x2, a_y1)
    box_a_corners[2].set(a_x2, a_y2)
    box_a_corners[3].set(a_x1, a_y2)

    box_b_corners = [Point() for i in range(5)]
    box_b_corners[0].set(b_x1, b_y1)
    box_b_corners[1].set(b_x2, b_y1)
    box_b_corners[2].set(b_x2, b_y2)
    box_b_corners[3].set(b_x1, b_y2)

    a_angle_cos = cos(a_angle)
    a_angle_sin = sin(a_angle)
    b_angle_cos = cos(b_angle)
    b_angle_sin = sin(b_angle)

    for k in range(0, 4):
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k])
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k])
    

    box_a_corners[4] = box_a_corners[0]
    box_b_corners[4] = box_b_corners[0]

    cross_points = [Point() for i in range(16)]
    poly_center = Point()
    cnt = 0
    flag = 0

    poly_center.set(0, 0)
    for i in range(0, 4):
        for j in range(0, 4):
            flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                                box_b_corners[j + 1], box_b_corners[j],
                                cross_points[cnt])
            if flag:
                poly_center = poly_center + cross_points[cnt]
                cnt += 1

    for k in range(0, 4):
        if (check_in_box2d(box_a, box_b_corners[k])):
            poly_center = poly_center + box_b_corners[k]
            cross_points[cnt] = box_b_corners[k]
            cnt += 1
        if (check_in_box2d(box_b, box_a_corners[k])):
            poly_center = poly_center + box_a_corners[k]
            cross_points[cnt] = box_a_corners[k]
            cnt += 1

    if cnt == 0:
        return 0.0
    poly_center.x /= cnt
    poly_center.y /= cnt

    temp = Point()
    for j in range(cnt - 1):
        for i in range(cnt - j - 1):
            if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)):
                temp = cross_points[i]
                cross_points[i] = cross_points[i + 1]
                cross_points[i + 1] = temp

    area = 0
    for k in range(0, cnt - 1):
        area += cross(cross_points[k] - cross_points[0],
                        cross_points[k + 1] - cross_points[0])

    return abs(area) / 2.0


def boxes_overlap_bev_gpu(boxes_a, boxes_b, ans_overlap):
    for i, box_a in enumerate(boxes_a):
        for j, box_b in enumerate(boxes_b):
            ans = box_overlap(box_a, box_b)
            ans_overlap[i][j] = ans
            ans_overlap[j][i] = ans


def iou_bev(box_a, box_b):
    sa = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    sb = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    s_overlap = box_overlap(box_a, box_b)
    return s_overlap / max(sa + sb - s_overlap, EPS)


def boxes_iou_bev_gpu(boxes_a, boxes_b, ans_iou):
    for i, box_a in enumerate(boxes_a):
        for j, box_b in enumerate(boxes_b):
            if (box_a == box_b).all():
                ans = 1.0
            else:
                ans = iou_bev(box_a, box_b)
            ans_iou[i][j] = ans
            ans_iou[j][i] = ans


def nms_gpu(boxes, keep, nms_overlap_thresh, device_id):

    idxs = torch.LongTensor(range(len(boxes)))
    to_reject = torch.LongTensor(range(len(boxes)))
    to_reject.fill_(1)

    for index, box in enumerate(boxes):
        print(f"{index=}")
        if not to_reject[index]:
            continue

        mask = torch.logical_and(
            idxs > index,
            to_reject
        )
        target_boxes = boxes[mask]
        ious = []

        for tg_box in target_boxes:
            print(f"{tg_box=}")
            ious.append(iou_bev(box, tg_box))
        print(f"{ious=}")

        rej = torch.where(torch.Tensor(ious) > nms_overlap_thresh)[0] + index + 1
        to_reject[rej] = 0
        print(f"{to_reject=}")
    
    to_save = torch.where(to_reject)[0]
    print(to_save)
    keep[:len(to_save)] = boxes[to_save]
    return len(to_save)
