import numpy as np
#from keras.models import Sequential, Model
#from keras.layers import Dense, Dropout, Input, Conv2D, BatchNormalization, LeakyReLU, MaxPool2D
from keras import backend as K
from PIL import Image


"""
confidence_loss = objects_loss + no_objects_loss
objects_loss = object_scale * detectors_mask * K.square(1 - pred_confidence)
no_objects_loss = no_object_scale * (1 - object_detections) * (1 - detectors_mask) * K.square(-pred_confidence)

classification_loss = class_scale * detectors_mask * K.square(matching_classes - pred_class_prob)

coordinates_loss = coordinates_scale * detectors_mask * K.square(matching_boxes - pred_boxes)
"""

def get_data(annotation_lines, input_shape):
    '''random preprocessing for real-time data augmentation'''
    imgs = []
    boxes = []
    for annotation_line in annotation_lines:
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        #print(box)

        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2

        image_data=0
        
        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image)/255.
        imgs.append(image_data)
        

        # correct boxes
        #box_data = np.zeros(5)
        if len(box)>0:
            box = box[0]
            # width
            box[[0,2]] = box[[0,2]]*scale + dx
            # height
            box[[1,3]] = box[[1,3]]*scale + dy
            # confidence
            box[4:] = 1
            #box_data[:len(box)] = box
            #print(box)
        boxes.append(box)
    return imgs, boxes

def get_one_data(annotation_line, input_shape):

    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    #print(box)
    # resize image
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = (w-nw)//2
    dy = (h-nh)//2
    image_data=0
    
    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image)/255.
    
    # correct boxes
    #box_data = np.zeros(5)
    if len(box)>0:
        box = box[0]
        # width
        box[[0,2]] = box[[0,2]]*scale + dx
        # height
        box[[1,3]] = box[[1,3]]*scale + dy
        # confidence
        box[4:] = 1
        #box_data[:len(box)] = box
        #print(box)
    return image_data, box

#print(line)
input_shape = (320, 240)
grid_shape = (20, 15)
batch_size = 16
"""
with open('./train.txt', 'r') as f:
    lines = f.readlines()
imgs, boxes = get_data(lines, input_shape)
print(len(imgs), len(boxes))

image = imgs
box = np.array(boxes).astype('float32')
obj_mask = np.zeros((len(boxes), *grid_shape))
box_xy = (box[..., 0:2] + box[..., 2:4]) // 2
box_wh = box[..., 2:4] - box[..., 0:2]
box[..., 0:2] = (box_xy / input_shape[::-1])
box[..., 2:4] = (box_wh / input_shape[::-1])

j = np.floor(box[..., 0] * grid_shape[1]).astype('int32')
i = np.floor(box[..., 1] * grid_shape[0]).astype('int32')
#print(i, j)
print(box.shape)
for b in range(len(boxes)):
    obj_mask[b, i[b], j[b]] = 1

nonobj_mask = 1 - obj_mask
#print(nonobj_mask[0])
print(obj_mask[0].shape)



# obj_mask, noobj_mask 

label = np.zeros((batch_size, grid_shape[0], grid_shape[1], 5))
print(label.shape)
label[0, i, j, :] = box[0, :]

print(label[..., 4:5].shape)
"""

def data_generator(annotation_path, batch_size, grid_shape=(20, 15)):
    with open(annotation_path, 'r') as f:
        annotation_lines = f.readlines()
    index = 0   
    num = len(annotation_lines)
    while True:
        imgs = []
        boxes = []
        for b in range(batch_size):
            if index == 0:
                np.random.shuffle(annotation_lines)
            img, one_box = get_one_data(annotation_lines[index], input_shape)
            imgs.append(img)
            boxes.append(one_box)
            index = (index+1) % num
        #get_data(annotation_lines[index:index + batch_size], input_shape)
        image = np.array(imgs)
        box = np.array(boxes).astype('float32')
        
        box_xy = (box[..., 0:2] + box[..., 2:4]) // 2
        box_wh = box[..., 2:4] - box[..., 0:2]
        box[..., 0:2] = (box_xy / input_shape[::-1])
        box[..., 2:4] = (box_wh / input_shape[::-1])

        j = np.floor(box[..., 0] * grid_shape[1]).astype('int32')
        i = np.floor(box[..., 1] * grid_shape[0]).astype('int32')
        label = np.zeros((batch_size, grid_shape[0], grid_shape[1], 5))
        label[0, i, j, :] = box[0, :]
        print(index, index+batch_size)

        yield [image, label], np.zeros(batch_size)
        #index = index + batch_size
        #if index > len(annotation_lines):
        #    return 'done'


#for _, data in data_generator("./train.txt", batch_size):
#    print(data)
    
def yolo_loss(args, batch_size=16, grid_shape=(20,15)):
    y_true = args[1]
    y_pred = args[0]
    print(y_pred.shape)
    obj_mask = y_true[..., 4]
    nonobj_mask = 1 - obj_mask
    #print(y_true)
    """
    j = np.floor(y_true[..., 0] * grid_shape[1]).astype('int32')
    i = np.floor(y_true[..., 1] * grid_shape[0]).astype('int32')
    #print(i, j)
    for b in range(batch_size):
        obj_mask[b, i[b], j[b]] = 1
    """
    
    #print(nonobj_mask[0])
    #print(obj_mask[0])

    lambda_coord = 5
    lambda_noobj = 0.5
    xy = obj_mask * lambda_coord * K.sum(K.square(y_pred[..., 0:2] - y_true[..., 0:2]), -1)
    wh = obj_mask * lambda_coord * K.sum(K.square(K.sqrt(y_pred[..., 2:4]) - K.sqrt(y_true[..., 2:4])), -1)
    conf = obj_mask * K.sum(K.square(y_pred[..., 4:] - y_true[..., 4:])) + nonobj_mask * lambda_noobj * K.sum(K.square(y_pred[..., 4:] - y_true[..., 4:]))
    loss = K.sum(xy) + K.sum(wh) + K.sum(conf)
    return loss


"""
def yolo_loss(y_true, y_pred):

    lambda_coord = 5
    lambda_noobj = 0.5
    xy = lambda_coord * K.sum(K.square(y_pred[..., 0:2] - y_true[..., 0:2]), -1)
    wh = lambda_coord * K.sum(K.square(K.sqrt(y_pred[..., 2:4]) - K.sqrt(y_true[..., 2:4])), -1)
    conf = K.sum(K.square(y_pred[..., 4:] - y_true[..., 4:])

    return xy + wh + conf


"""

"""
output = [batch, grid_size[0], grid_size[1], 5]
label  = [batch, grid_size[0], grid_size[1], 5]

loss = 
lambda * obj_mask * K.sum(K.square(output[..., 0:2] - label[..., 0:2]), -1)
lambda * obj_mask * K.sum(K.square(K.sqrt(output[..., 2:4]) - K.sqrt(label[..., 2:4])), -1)
lambda * obj_mask * K.sum(K.square(output[..., 4:] - label[..., 4:])

np.sum(np.square(b-a), -1)
np.sum(np.square(np.sqrt(b)-np.sqrt(a)), -1)

"""