# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from munch import DefaultMunch
import numpy as np
import tensorflow as tf


class ObjectDetectionMetricsData:
    """
    This class is used to store batches of groundtruth (GT) labels and detections,
    which are used to calculate the mAP metrics.  
    """
    
    def __init__(self, num_labels, num_boxes, dataset_size, name=None, **kwargs):
    
        self.num_labels = num_labels
        self.num_boxes = num_boxes
        
        # Total size of the data to store, equal to the sum 
        # of the sizes of all the batches to be received
        self.dataset_size = dataset_size
        
        # The index where to store a new batch of GT labels and detections
        # in the tensors. When a new batch of data is received, it is saved in
        # the tensors from index update_index to update_index + batch_size.
        self.update_index = tf.Variable(0, trainable=False, dtype=tf.int32)

        # Groundtruth labels
        self.gt_labels_ds = tf.Variable(tf.zeros([dataset_size, num_labels, 5]), trainable=False, dtype=tf.float32)

        # Detections boxes, scores and classes
        self.boxes_ds = tf.Variable(tf.zeros([dataset_size, num_boxes, 4]), trainable=False, dtype=tf.float32)
        self.scores_ds = tf.Variable(tf.zeros([dataset_size, num_boxes]), trainable=False, dtype=tf.float32)
        self.classes_ds = tf.Variable(tf.zeros([dataset_size, num_boxes]), trainable=False, dtype=tf.float32)


    def add_data(self, gt_labels, boxes, scores, classes):
        """
        Stores a new batch of grountruth labels and detections data.
        """
        
        # Create indices to update the tensors that store the data
        batch_size = tf.shape(gt_labels)[0]
        indices = tf.range(self.update_index, self.update_index + batch_size)
        indices = tf.expand_dims(indices, axis=1)
        
        # Store the batch of GT labels
        self.gt_labels_ds.assign(tf.tensor_scatter_nd_update(self.gt_labels_ds, indices, gt_labels))
        
        # Store the batch of detections
        self.boxes_ds.assign(tf.tensor_scatter_nd_update(self.boxes_ds, indices, boxes))
        self.scores_ds.assign(tf.tensor_scatter_nd_update(self.scores_ds, indices, scores))
        self.classes_ds.assign(tf.tensor_scatter_nd_update(self.classes_ds, indices, classes))
        
        # Update the index to the end of the update we just made
        # Reset it to 0 if all the batches of data have been received
        self.update_index.assign((self.update_index + batch_size) % self.dataset_size)


    def get_data(self):
        """
        Returns the GT labels and detections that have been stored.

        Image numbers are assigned to all GT labels and detections
        so that the GT labels and detections that belong to the same
        image can be retreived.
        
        GT labels may contain [0,0,0,0,0] labels that are used to pad
        tensors. Similarly, detection boxes may contain [0,0,0,0] boxes. 
        These dummy labels and boxes are filtered.
        """
    
        # Output groudtruth box: [image_number, class, x1, y1, x2, y2]
        image_nbrs = tf.repeat(tf.range(self.dataset_size), self.num_labels)
        image_nbrs = tf.reshape(image_nbrs, [self.dataset_size, self.num_labels, 1])
        image_nbrs = tf.cast(image_nbrs, tf.float32)
        
        groundtruths = tf.concat([image_nbrs, self.gt_labels_ds], axis=-1)
        
        # Output detection: [image_number, class, score, x1, y1, x2, y2]
        image_nbrs = tf.repeat(tf.range(self.dataset_size), self.num_boxes)
        image_nbrs = tf.reshape(image_nbrs, [self.dataset_size, self.num_boxes, 1])
        image_nbrs = tf.cast(image_nbrs, tf.float32)

        classes = tf.expand_dims(self.classes_ds, axis=-1)
        scores = tf.expand_dims(self.scores_ds, axis=-1)
        detections = tf.concat([image_nbrs, classes, scores, self.boxes_ds], axis=-1)
        
        # Filter padding groundtruth labels
        coords_sum = tf.math.reduce_sum(groundtruths[..., 2:], axis=-1)
        indices = tf.where(coords_sum > 0)
        groundtruths = tf.gather_nd(groundtruths, indices)

       # Filter padding detection boxes
        coords_sum = tf.math.reduce_sum(detections[..., 3:], axis=-1)
        indices = tf.where(coords_sum > 0)
        detections = tf.gather_nd(detections, indices)

        return groundtruths, detections
        
        
    def reset(self):  
    
        """
        Reset the tensors that store the GT labels and detections data.
        """
        
        # Reset the index that points to the location in the tensors
        # of the next batch of data to be stored
        self.update_index.assign(0)

        # Reset the GT labels
        self.gt_labels_ds.assign(tf.zeros([self.dataset_size, self.num_labels, 5], dtype=tf.float32))

        # Reset the detections
        self.boxes_ds.assign(tf.zeros([self.dataset_size, self.num_boxes, 4], dtype=tf.float32))
        self.scores_ds.assign(tf.zeros([self.dataset_size, self.num_boxes], dtype=tf.float32))
        self.classes_ds.assign(tf.zeros([self.dataset_size, self.num_boxes], dtype=tf.float32))


def calculate_iou(boxA, boxB):
    """
    Calculate the IOU between two bounding boxes
    Coordinates must be in (x1, y1, x2, y2) format.
    """
    
    boxA_x1, boxA_y1, boxA_x2, boxA_y2 = boxA
    boxB_x1, boxB_y1, boxB_x2, boxB_y2 = boxB
    
    # If the two boxes don't intersect, the IOU is 0.
    if boxA_x1 > boxB_x2 or boxA_x2 < boxB_x1:
        return 0.
    if boxA_y1 > boxB_y2 or boxA_y2 < boxB_y1:
        return 0.
        
    # Calculate the coordinates of diagonally opposite
    # corners of the intersection of boxA and boxB
    inter_x1 = max(boxA_x1, boxB_x1)
    inter_y1 = max(boxA_y1, boxB_y1)
    inter_x2 = min(boxA_x2, boxB_x2)
    inter_y2 = min(boxA_y2, boxB_y2)
    
    inter_x = inter_x2 - inter_x1 + 1
    inter_y = inter_y2 - inter_y1 + 1
    inter_area = max(inter_x, 0.) * max(inter_y, 0.)
    
    boxA_area = (boxA_x2 - boxA_x1 + 1) * (boxA_y2 - boxA_y1 + 1)
    boxB_area = (boxB_x2 - boxB_x1 + 1) * (boxB_y2 - boxB_y1 + 1)

    union_area = boxA_area + boxB_area - inter_area

    iou = inter_area / union_area if union_area > 0. else 0.

    return iou


def calculate_average_metrics(metrics):
    """
    Calculate average precision, recall and AP
    """
    mpre = np.mean([v.pre for v in metrics.values()])
    mrec = np.mean([v.rec for v in metrics.values()])
    mAP = np.mean([v.ap for v in metrics.values()])
    
    return mpre, mrec, mAP
    

def calculate_objdet_metrics(groundtruths_ds, detections_ds, iou_threshold=None, averages_only=False):
    """
    Calculate precision, recall and AP for each class
    
    Arguments:
    ---------
        groundtruths_ds:
            Groundtruth labels of the entire dataset, a tensor with shape [num_gt, 6]
            Items: [image_number, class, x1, y1, x2, y2]
            
        detections_ds:
            Detections of the entire dataset, a tensor with shape [num_detections, 7]
            Items: [image_number, class, score, x1, y1, x2, y2]
            
        iou_threshold:
            IOU threshold to use to classify detections as true positives or false positives.
            A detection box is a true positive if it has an IOU with one of the groundtruth boxes
            that is greater than or equal to the threshold.
     
    Groundtruth labels and detections are associated using the image numbers.
    All the labels and detections that have the same image number belong
    to the same image.

    Returns:
    -------
        A dictionary
    
    """
    
    # Convert the input tensors to lists
    groundtruths_ds = groundtruths_ds.numpy().tolist()
    detections_ds = detections_ds.numpy().tolist()
    
    # Get the class numbers used in the dataset
    classes = set([g[1] for g in groundtruths_ds])
    classes = sorted(list(classes))
    
    metrics = {}
    for c in classes:   
            
        # Get the groundtruths and detections for current class c
        groundtruths = [g for g in groundtruths_ds if g[1] == c]
        detections = [d for d in detections_ds if d[1] == c]
        
        # Sort detections by decreasing confidence
        detections = sorted(detections, key=lambda conf: conf[2], reverse=True)

        # Create a dictionary 'image_gts' for fast access to the GTs
        # of a given image using the image number as a key.
        # Also create a dictionary 'matched_gts' to keep track
        # of GTs that have already been matched with a detection.
        image_gts = {}
        matched_gts = {}
        for gt in groundtruths:
            image_nbr = int(gt[0])
            image_gts[image_nbr] = image_gts.get(image_nbr, []) + [gt]
            matched_gts[image_nbr] = matched_gts.get(image_nbr, []) + [0]

        # Initialize true positives and false positives
        TP = np.zeros(len(detections), dtype=np.float32)
        FP = np.zeros(len(detections), dtype=np.float32)
        
        # Loop through detections
        for i, det in enumerate(detections):
        
            # Get the GTs from the same image as the detection
            image_nbr = int(det[0])
            gts = image_gts.get(image_nbr, [])
            matched = matched_gts.get(image_nbr, [])
            
            # Calculate the IOUs of the detection with all the GTs
            # Look for the maximum IOU value and the corresponding GT
            iou_max = -1
            for k, gt in enumerate(gts):
                iou = calculate_iou(det[3:], gt[2:])
                if iou > iou_max:
                    iou_max = iou
                    gt_max = k

            # Classify the detection as true positive or false positive
            if iou_max >= iou_threshold:
                if matched[gt_max] == 0:
                    TP[i] = 1
                    # Flag the GT as already matched
                    matched[gt_max] = 1
                    matched_gts[image_nbr] = matched
                else:
                    FP[i] = 1
            else:
                FP[i] = 1
        
        # Compute precision and recall    
        npos = len(groundtruths)        
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / npos if len(acc_TP) > 0 and npos > 0 else [0.]
        pre = acc_TP / (acc_FP + acc_TP) if len(acc_TP) > 0 else [0.]
        
        # Interpolate the precision
        ipre = np.concatenate(([0.], pre, [0.]))
        for i in range(len(ipre) - 1, 0, -1):
            ipre[i - 1] = max(ipre[i - 1], ipre[i])
                
        # Find the indices where recall changes
        ii = []
        irec = np.concatenate(([0.], rec, [1.]))
        for i in range(len(irec) - 1):
            if irec[1 + i] != irec[i]:
                ii.append(i + 1)

        # Calculate the average precision (AUC)
        ap = 0
        for i in ii:
            ap = ap + np.sum((irec[i] - irec[i - 1]) * ipre[i])

        # Record the class metrics
        metrics[int(c)] = {
            'pre': np.mean(pre),
            'rec': np.mean(rec),
            'ap': ap,
            'interpolated_precision': ipre,
            'interpolated_recall': irec,
        }

    metrics = DefaultMunch.fromDict(metrics)
    
    if averages_only:
        return calculate_average_metrics(metrics)
    else:
        return metrics