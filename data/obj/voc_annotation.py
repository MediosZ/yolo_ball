import xml.etree.ElementTree as ET
import glob, os

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["ball"]


def convert_annotation(image_id, list_file):
    in_file = open('%s.xml'% image_id)
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = os.getcwd()
"""
for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()
    """

current_dir = os.path.dirname(os.path.abspath(__file__))
list_file = open('train.txt', 'w')

for pathAndFilename in glob.iglob(os.path.join(current_dir, '*.png')):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    list_file.write('data/obj/' + title + ext)
    #print('data/obj/' + title + ext)
    convert_annotation(title, list_file)
    list_file.write('\n')


list_file.close()
