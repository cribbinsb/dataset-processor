import os
import yaml
import cv2
import math
import copy
import stuff
import base64

import numpy as np
from datetime import datetime
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.loaders.coco_loader import CocoLoader
from src.loaders.openimages_loader import OpenImagesLoader
from src.loaders.o365_loader import O365Loader
from src.loaders.widerface_loader import WiderfaceLoader
from src.loaders.roboflow_loader import RoboflowLoader
from src.loaders.weapondetection_loader import WeaponDetectionLoader
from src.loaders.widerattributes_loader import WiderAttributesLoader
#from src.llm.llm_ollama import LLMOllama
from src.llm.llm_openai import LLMOpenAI
from src.llm.llm_anthropic import LLMAnthropic
from src.llm.llm_mobileclip import LLMMobileCLIP

def mldata_folder():
    d=os.environ.get('MLDATA_LOCATION')
    if d==None:
        return "/mldata"
    return d

def unique_dataset_name(dataset_name):
    ver=0
    while True:
        name=dataset_name
        if not os.path.isdir(mldata_folder()+"/"+name):
            break
        name+="-"+datetime.today().strftime('%y%m%d')
        if ver!=0:
            name+="-v"+str(ver)
        if not os.path.isdir(mldata_folder()+"/"+name):
            break
        ver+=1
    return name

def name_from_file(x):
    ext=""
    if isinstance(x, str):
        if ":" in x:
            t=x.split(":")
            x=t[0]
            ext+=t[1]+" "
        if x.endswith(".engine"):
            ext+="TRT "
        if len(ext)>0:
            ext="("+ext[:-1]+")"
    return os.path.splitext(os.path.basename(x))[0]+ext
     
def get_all_dataset_files(dataset, task="val", file_key="both"):
    """
    'Load' a yaml dataset and return arrays of all the corresponding image and label files

    Args:
        dataset: path of yaml file
        task: train or val
        file_key: if 'both' returns img/label where both exist, if 'image' returns all images, if 'label' returns all labels
        
    Returns:
        list of dict, each containing "label" and "image" keys with paths
        list of class names
    """
    
    assert(file_key=="both" or file_key=="image" or file_key=="label")

    with open(dataset) as stream:
        try:
            ds=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None, None, None, None

    if not "names" in ds:
        print(f"Could not class names in dataset {dataset} yaml")
        return None, None, None, None
    
    names=ds["names"]

    if "dataset_name" in ds:
        dataset_name=ds["dataset_name"]
    else:
        dataset_name=name_from_file(dataset)

    attributes=None
    if "attributes" in ds:
       attributes=ds["attributes"]

    if not task in ds:
        print(f"Could not find {task} in dataset {dataset} yaml")
        return None, None, None, None
    
    ds[task]=str(ds[task])

    val=ds[task].strip()

    if val.startswith('[') and val.endswith(']'):
        val=val[1:-1].split(",")
        val=[v.strip() for v in val]
    else:
        val=[val]

    for i,v in enumerate(val):
        if v.startswith('"') and v.endswith('"'):
            v=v[1:-1]
        if v.startswith("'") and v.endswith("'"):
            v=v[1:-1]
        val[i]=v

    if "path" in ds:
        path=ds["path"]
        if path==".":
            path=os.path.dirname(dataset)
        val=[os.path.join(path, v) for v in val]

    files=[]
    
    for v in val:
        for x in ["/images","/labels"]:
            if v.endswith(x):
                v=v[:-len(x)]
        img_path=os.path.join(v, "images")
        label_path=os.path.join(v, "labels")
        if file_key=="image" or file_key=="both":
            if os.path.isdir(img_path):
                imgs=sorted(os.listdir(img_path))
                for i in imgs:
                    l=os.path.splitext(i)[0]+".txt"
                    img=os.path.join(img_path, i)
                    lab=os.path.join(label_path,l)
                    f={}
                    if os.path.isfile(img) and os.path.isfile(lab):
                        f={"image":img, "label":lab}
                        files.append(f)
                    elif file_key=="image" and os.path.isfile(img):
                        f={"image":img, "label":None}
                        files.append(f)
        else:
            if os.path.isdir(label_path):
                labels=sorted(os.listdir(label_path))
                for l in labels:
                    i=os.path.splitext(i)[0]+".jpg"
                    img=os.path.join(img_path, i)
                    lab=os.path.join(label_path,l)
                    if os.path.isfile(img) and os.path.isfile(lab):
                        f={"image":img, "label":lab}
                        files.append(f)
                    elif file_key=="label" and os.path.isfile(lab):
                        f={"image":None, "label":lab}
                        files.append(f)

    return files, names, dataset_name, attributes

def make_dataset_yaml(name, config_name=None,
                      class_names=None, face_kp=True, pose_kp=True,
                      facepose_kp=False, 
                      attributes=None,
                      poseattr=None):
    """
    Write out a basic dataset yaml description
    """
    path=mldata_folder()+"/"+name

    txt="# autogenerated dataset YAML file for "+name+"\n"
    txt+="# created: "+datetime.now().strftime("%d/%m/%Y %H:%M:%S")+"\n"
    txt+="dataset_name: "+os.path.split(name)[1]+"\n"
    if config_name is not None:
        txt+="config_name: "+config_name+"\n"
    txt+="path: "+path+"\n"
    txt+="train: train/images\n"
    txt+="val: val/images\n"
    txt+="nc: "+str(len(class_names))+"\n"
    num_kpt=0
    flip_idx=[]
    if face_kp and pose_kp:
        num_kpt=22
        flip_idx=[1, 0, 2, 4, 3, 5, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20]
    elif face_kp:
        num_kpt=5
        flip_idx=[1, 0, 2, 4, 3]
    elif pose_kp:
        num_kpt=17
        flip_idx=[0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    elif facepose_kp:
        num_kpt=19
        flip_idx=[0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]
    if poseattr is not None:
        for i in range(len(poseattr)):
            flip_idx.append(num_kpt+i)
        num_kpt+=len(poseattr)
    if num_kpt!=0:
        txt+=f"kpt_shape: [{num_kpt}, 3]\n"
        txt+=f"flip_idx: {flip_idx}\n"
    txt+="names:\n"
    for c in class_names:
        txt+="    - "+c+"\n"
    if poseattr is not None:
        txt+="poseattr:\n"
        for p in poseattr:
            txt+="    - "+p+"\n"
    if attributes is not None:
        txt+="attributes:\n"
        for a in attributes:
            txt+="    - "+a+"\n"
    
    stuff.makedir(path)
    stuff.makedir(path+"/train/images")
    stuff.makedir(path+"/train/labels")
    stuff.makedir(path+"/val/images")
    stuff.makedir(path+"/val/labels")
    yaml_file_name=path+"/dataset.yaml"
    with open(yaml_file_name, 'w') as file:
        file.write(txt)
    return yaml_file_name

def load_ground_truth_labels(label_file, num_poseattr=None):
    """
    load one label file

    Args:
        path of label .txt file
        
    Returns:
        list of boxes co-ords [x1,x2,y1,y2]
        list of box classes
    """
    if label_file==None:
        return None
    try:
        if not os.path.isfile(label_file):
            return None
    except:
        print("Exception loading gt "+str(label_file))
        return None
    
    out=[]
    with open(label_file, 'r') as f:
        boxes=[]
        classes=[]
        for line in f:
            ss=line.strip().split()
            attrs=None
            if "A" in ss:
                index=ss.index("A")
                attrs=[float(x) for x in ss[index+1:]]
                ss=ss[0:index]
            
            vals=[float(x) for x in ss]
        
            box=[vals[1]-0.5*vals[3], vals[2]-0.5*vals[4], vals[1]+0.5*vals[3], vals[2]+0.5*vals[4]]
            cls=int(vals[0])
            gt={"box":box,
                "class":cls,
                "confidence":1.0}
            
            if attrs is not None:
                gt["attrs"]=attrs

            if num_poseattr is not None:
                l=len(vals)
                gt["poseattr"]=[0]*num_poseattr
                for k in range(num_poseattr):
                    gt["poseattr"][k]=vals[l-num_poseattr*3+3*k+2]
                    assert vals[l-num_poseattr*3+3*k+0]<0.01
                    assert vals[l-num_poseattr*3+3*k+1]<0.01
            
            if len(vals)==20 or len(vals)==71:
                gt["face_points"]=vals[5:20]
            if len(vals)==56:
                gt["pose_points"]=vals[5:56]
            if len(vals)==62:
                gt["facepose_points"]=vals[5:62]
            if len(vals)==71:
                gt["pose_points"]=vals[20:71]
            out.append(gt)
            
    return out

def load_cached_detections(model_path, dataset, f):
    """
    load_cached_detections

    Args:
        path to model .pt file
        path to dataset yaml file
        path to where a label .txt file would be if it existed
        
    Returns:
        flat list of boxes co-ords x1,x2,y1,y2
        list of box classes
        list of confidences
    """
    model=name_from_file(model_path)
    dataset=name_from_file(dataset)
    f=os.path.basename(f)
    path=os.path.join("/mldata/detections",dataset,model,f)

    if not os.path.isfile(path):
        return None, None, None
    
    boxes=[]
    classes=[]
    confidences=[]

    with open(path, 'r') as f:
        for line in f:
            vals=[float(x) for x in line.strip().split()]
            box=[vals[1]-0.5*vals[3], vals[2]-0.5*vals[4], vals[1]+0.5*vals[3], vals[2]+0.5*vals[4]]
            boxes.append(box)
            classes.append(vals[0])
            confidences.append(vals[5])
    return boxes, classes, confidences

def is_large(x):
    return x["box"][2]-x["box"][0]>=0.1

def has_face_points(x):
    """
    Return True if the annotation has non-trivial face points (i.e. not all 0)
    """
    t=0
    if "face_points" in x:
        for i in range(5):
            t+=x["face_points"][3*i+2]
    if "facepose_points" in x:
        for i in [0,1,2,17,18]:
            t+=x["facepose_points"][3*i+2]
    if t<0.1:
        return False
    return True

def has_pose_points(x):
    """
    Return True if the annotation has non-trivial pose points (i.e. not all 0)
    """
    t=0
    if "pose_points" in x:
        for i in range(17):
            t+=x["pose_points"][3*i+2]
    if "facepose_points" in  x:
        for i in range(3,17):
            t+=x["facepose_points"][3*i+2]
    if t<0.1:
        return False
    return True

def better_annotation(a1, a2):
    """
    return True if a1 is 'better' than a2
    better is currently the one that has pose, face points if the other doesn't else the largest
    """
    a1_pp=has_pose_points(a1)
    a2_pp=has_pose_points(a2)
    if a1_pp!=a2_pp:
        return a1_pp
    a1_fp=has_face_points(a1)
    a2_fp=has_face_points(a2)
    if a1_fp!=a2_fp:
        return a1_fp
    return stuff.box_a(a1["box"])>stuff.box_a(a2["box"])
    
def dedup_gt(gts, iou_thr=0.6):
    """
    Remove multiple overlapping instances of the same class
    This can happen because things that are different in the original dataset may be mapped to the same label
    e.g. head,face or boy,person
    """
    for i,g in enumerate(gts):
        for j,g2 in enumerate(gts):
            if i==j or g2["confidence"]==0 or g["confidence"]==0:
                continue
            if g["class"]==g2["class"]:
                if stuff.box_iou(g["box"], g2["box"])>iou_thr:
                    if better_annotation(g, g2): # delete the 'worst' one
                        g2["confidence"]=0
                    else:
                        g["confidence"]=0
    out=[]
    for g in gts:
        if g["confidence"]>0:
            out.append(g)
    return out

def best_iou_match(x, to_check):
    """
    check against list and find match of same class with largest IOU. Retun triple iou, matching item, item index
    """
    best_iou=0
    best_g=None
    best_i=-1
    for i,g in enumerate(to_check):
        if x["class"]==g["class"]:
            iou=stuff.box_iou(x["box"],g["box"])
            if iou>best_iou:
                best_iou=iou
                best_g=g
                best_i=i
    return best_iou, best_g, best_i

def kp_iou(kp_gt, kp_det, s, num_pt):
    """
    Computes the keypoint iou between two boxes

    Args:
        kpgt, gpdet: list [x,y,conf]*npoint
        s: GT box area
    Returns:
        float iou 0-1
    """
    # https://learnopencv.com/object-keypoint-similarity/
    if num_pt==5:
        scales=[0.025, 0.025, 0.026, 0.025, 0.025]
    elif num_pt==17:
        scales=[0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
    elif num_pt==22:
        scales=[0.025, 0.025, 0.026, 0.025, 0.025, 0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
    else:
        scales=[0.025]*num_pt

    scales=[x*2 for x in scales] # scale=2*sigma

    ss=s*0.53 # approximation of shape area from box area
    num=0
    denom=0
    for i in range(num_pt):
        if kp_gt[i*3+2]>0.3: # is point labelled
            dx=kp_gt[i*3+0]-kp_det[i*3+0]
            dy=kp_gt[i*3+1]-kp_det[i*3+1]
            num+=math.exp(-(dx*dx+dy*dy)/(2.0*ss*scales[i]*scales[i]+1e-7))
            denom+=1.0
    iou=num/(denom+1e-7)
    return iou

def kp_iou2(kp_gt, kp_det, s, num_pt):
    """
    Computes the keypoint iou between two boxes

    Args:
        kpgt, gpdet: list [x,y,conf]*npoint
        s: GT box area
    Returns:
        float iou 0-1
    """
    # https://learnopencv.com/object-keypoint-similarity/
    if num_pt==5:
        scales=[0.025, 0.025, 0.026, 0.025, 0.025]
    elif num_pt==17:
        scales=[0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
    elif num_pt==22:
        scales=[0.025, 0.025, 0.026, 0.025, 0.025, 0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
    else:
        scales=[0.025]*num_pt

    scales=[x*2 for x in scales] # scale=2*sigma

    ss=s*0.53 # approximation of shape area from box area
    num=0
    denom=0
    for i in range(num_pt):
        if kp_gt[i*3+2]>0.01 and kp_det[i*3+2]>0.01: # BOTH points labelled
            dx=kp_gt[i*3+0]-kp_det[i*3+0]
            dy=kp_gt[i*3+1]-kp_det[i*3+1]
            num+=math.exp(-(dx*dx+dy*dy)/(2.0*ss*scales[i]*scales[i]+1e-7))
            denom+=1.0
    iou=num/(denom+1e-7)
    return iou

def match_boxes(dets, gts, iou_thr=0.5):
    """
    Matches predictions to ground truth values

    Args:
        det_boxes, gt_boxes list of [x1,y1,x2,y2] in xyxy format
        det_classes, gt_classes: class indexes
        iou_thr: min IOU to count as a match
        note: det boxes/classes should be pre-sorted in descending confidence order
        
    Returns:
        det_matched, gt_matched: corresponding indexes of matches, -1 means for no match
    """
    gt_matched=[-1]*len(gts)
    det_matched=[-1]*len(dets)
    for j,_ in enumerate(dets):
        for i,_ in enumerate(gts):
            if gt_matched[i]==-1 and gts[i]['class']==dets[j]['class'] and stuff.box_iou(gts[i]['box'], dets[j]['box'])>iou_thr:
                gt_matched[i]=j
                det_matched[j]=i
                break
    return det_matched, gt_matched

def match_boxes_kp(dets, gts, iou_thr=0.5, face_class=None, person_class=None):
    """
    Match detections to GTs using keypoint IOU
    """
    gt_matched=[-1]*len(gts)
    det_matched=[-1]*len(dets)
    for j,_ in enumerate(dets):
        for i,_ in enumerate(gts):
            if gt_matched[i]==-1 and gts[i]['class']==dets[j]['class']:
                a=stuff.box_a(gts[i]['box'])
                if gts[i]['class']==face_class and 'face_points' in gts[i]:
                    #biou=stuff.box_iou(gts[i]['box'], dets[j]['box'])

                    #fp1=[0.095, 0.152, 1.0, 0.103, 0.154, 1.0, 0.098, 0.158, 1.0, 0.094, 0.162, 1.0, 0.102, 0.164, 1.0]
                    #fp2=[0.09501953423023224, 0.15155945718288422, 0.990234375, 0.10390625149011612, 0.1520467847585678, 0.98974609375, 0.09921874850988388, 0.15789473056793213, 0.990234375, 0.09526367485523224, 0.16252437233924866, 0.9912109375, 0.10234375298023224, 0.16301169991493225, 0.99072265625]
                    #fiou=kp_iou(fp1, fp2, a, 5)

                    #fiou=kp_iou(gts[i]['face_points'], dets[j]['face_points'], a, 5)
                    #if (biou>0.8):
                    #    print(fp1)
                    #    print(fp2)    
                    #    print(f" {biou} {fiou} {a}")
                    
                    if 'face_points' in dets[j] and kp_iou(gts[i]['face_points'], dets[j]['face_points'], a, 5)>iou_thr:
                        gt_matched[i]=j
                        det_matched[j]=i
                        break
                if gts[i]['class']==person_class and 'pose_points' in gts[i]:
                    if 'pose_points' in dets[j] and kp_iou(gts[i]['pose_points'], dets[j]['pose_points'], a, 17)>iou_thr:
                        gt_matched[i]=j
                        det_matched[j]=i
                        break
    return det_matched, gt_matched

def default_match(det, gt, context):
    if det["class"]!=gt["class"]:
        return 0
    return stuff.box_iou(det["box"], gt["box"])

def match_boxes_greedy(dets, gts, mfn=default_match, mfn_context=None):
    n_gts=len(gts)
    n_dets=len(dets)

    if n_gts==0 or n_dets==0:
        return [], []

    gt_matched=[False]*n_gts
    out_det_index=[]
    out_gt_index=[]
    for i,det in enumerate(dets):
        best_v=0
        best_match=None
        for j,gt in enumerate(gts):
            v=mfn(det,gt,mfn_context)
            if gt_matched[j]==False and v>best_v:
                best_v=v
                best_match=j
        if best_match is not None:
            gt_matched[best_match]=True
            out_det_index.append(i)
            out_gt_index.append(best_match)
    return out_det_index, out_gt_index

def match_boxes_lsa(dets, gts, mfn=default_match, mfn_context=None):
    n_gts=len(gts)
    n_dets=len(dets)

    if n_gts==0 or n_dets==0:
        return [], []
    
    costs = [[0 for x in range(n_dets)] for y in range(n_gts)]
    for ii in range(n_gts):
        for jj in range(n_dets):
            costs[ii][jj]=mfn(dets[jj], gts[ii], mfn_context)

    gt_ind, det_ind = linear_sum_assignment(np.array(costs), maximize=True)
    return det_ind, gt_ind

def append_comments(fn, x):
    x="#"+x
    x.replace('\n', '\n#')
    if not os.path.isfile(fn):
        print(f"Error: file {fn} does not exist")
        return 
    with open(fn, "a") as f:
        f.write(x+"\n")

def dataset_copy_comments(src, dest):
    with open(src, 'r') as file:
        for line in file:
            if line.startswith("#"):
                append_comments(dest, line[1:].strip())       

def attribute_text(a, attributes):
    text=""
    if "attrs" in a:
        l=list(range(len(a["attrs"])))
        l=[x for _, x in sorted(zip(a["attrs"], l), reverse=True)]
        for i in l:
            attr=a["attrs"][i]
            if attr>0.1:
                name=attributes[i]
                if name.startswith("person:"):
                    name=name[len("person:"):]
                text+=f"{name}={attr:0.2f}\n"
    return text

def add_pose_points_to_gts(dets, gts, iou_thr=0.6, min_sz=0.5*0.5, class_names=None):
    """
    Given existing GTs and detections from a pose point detector, add the pose values into the GTs
    does this by matching the box to find a good box in the GTs and copying the pose points to that box
    -i.e. assumption is the GTs already have full and complete labelling for the person boxes
    """
    person_class=class_names.index("person")
    added=0

    if True:
        det_ind, gt_ind=match_boxes_lsa(dets, gts)

        for i,_ in enumerate(gt_ind):
            gt=gts[gt_ind[i]]
            det=dets[det_ind[i]]
            iou=stuff.box_iou(gt["box"],det["box"])
            if det["class"]==person_class and gt["class"]==person_class and iou>iou_thr:
                if has_pose_points(gt) is False:
                    if "pose_points" in det:
                        gt["pose_points"]=det["pose_points"]
                    if "facepose_points" in det:
                        gt["facepose_points"]=det["facepose_points"]
                    added+=1
    return added

def face_match_function(f, p, context):
    if p["class"]!=context["person_class"]:
        return 0
    if f["class"]!=context["face_class"]:
        return 0
    # face must be not too small relative to body width
    if stuff.box_w(f["box"])<0.10*stuff.box_w(p["box"]):
        return 0
    # top of face has to be in top half of body
    if f["box"][1]>(0.5*p["box"][1]+0.5*p["box"][3]):
        return 0
    # face must not be too small relative to body
    if stuff.box_i(f["box"], p["box"])/(stuff.box_a(p["box"])+1e-10)<0.005:
        return 0
    ioa=stuff.box_i(f["box"], p["box"])/(stuff.box_a(f["box"])+1e-10)
    return ioa+f["confidence"]/100.0+p["confidence"]/10.0

def filter_faces_by_persons(faces, persons, ioa_thr=0.9, class_names=None):
    assert class_names is not None
    if not "person" in class_names:
        return faces # can't do any filtering

    assert "face" in class_names
    face_class=class_names.index("face")
    person_class=class_names.index("person")
    context={"person_class":person_class, "face_class":face_class}
    f_ind, p_ind=match_boxes_lsa(faces, persons, mfn=face_match_function, mfn_context=context)
    ret=[]
    for i,_ in enumerate(f_ind):
        p=persons[p_ind[i]]
        f=faces[f_ind[i]]
        ioa=stuff.box_i(f["box"], p["box"])/(stuff.box_a(f["box"])+1e-10)
        if ioa>ioa_thr:
            ret.append(f)
    return ret

def filter_persons_by_faces(dets, ioa_thr=0.9, add_anyway_conf=0.5, class_names=None):
    assert class_names is not None
    assert "person" in class_names
    assert "face" in class_names

    face_class=class_names.index("face")
    person_class=class_names.index("person")

    context={"person_class":person_class, "face_class":face_class}

    persons=[]
    faces=[]
    other=[]
    for d in dets:
        if d["class"]==person_class:
            persons.append(d)
        elif d["class"]==face_class:
            faces.append(d)
        else:
            other.append(d)

    f_ind, p_ind=match_boxes_lsa(faces, persons, mfn=face_match_function, mfn_context=context)

    for i,_ in enumerate(f_ind):
        p=persons[p_ind[i]]
        f=faces[f_ind[i]]
        ioa=stuff.box_i(f["box"], p["box"])/(stuff.box_a(f["box"])+1e-10)
        if ioa>ioa_thr:
            p["face_filter_ok"]=True

    persons_out=[]
    for p in persons:
        if "face_filter_ok" in p or p["confidence"]>add_anyway_conf:
            include=True
            for p2 in persons_out:
                if stuff.box_iou(p["box"],p2["box"])>0.5:
                    include=False
            if include:
                persons_out.append(p)

    return persons_out+faces+other

def sstr(x):
    if x==int(x):
        return str(int(x))+" "
    return "{:4.3f}  ".format(x)

def write_annotations(an, 
                      include_face=True, 
                      include_pose=False, 
                      include_facepose=False, 
                      include_attrs=True,
                      include_poseattr=None):
    an_txt=""
    for a in an:
        b=a["box"]
        cx=(b[2]+b[0])*0.5
        cy=(b[3]+b[1])*0.5
        w=b[2]-b[0]
        h=b[3]-b[1]
        an_txt+=str(a["class"])+" {0:.4f} {1:.4f} {2:.4f} {3:.4f} ".format(cx,cy,w,h)

        if include_facepose:
            assert(include_pose==False)
            assert(include_face==False)
            if "facepose_points" in a:
                fpp=a["facepose_points"]
            else:
                fpp=[0]*19*3
            for i in range(19*3):
                an_txt+=sstr(stuff.clip01(fpp[i]))
        if include_face:
            if "face_points" in a:
                fp=a["face_points"]
            else:
                fp=[0]*15
            for i in range(5*3):
                #if fp[i]<0 or fp[i]>1.0:
                #    print(f"Warning: face point {i} out of range {fp[i]}")
                an_txt+=sstr(stuff.clip01(fp[i]))
        if include_pose:
            if "pose_points" in a:
                pp=a["pose_points"]
            else:
                pp=[0]*17*3
            for i in range(17*3):
                if pp[i]<0 or pp[i]>1.0:
                    print(f"Warning: pose point {i} out of range {pp[i]}")
                an_txt+=sstr(stuff.clip01(pp[i]))
        if include_attrs:
            if "attrs" in a and a["attrs"] is not None:
                an_txt+="A "
                for i in a["attrs"]:
                    an_txt+=sstr(stuff.clip01(i))
        if include_poseattr is not None:
            if "poseattr" in a:
                assert(len(a["poseattr"])==len(include_poseattr))
            else:
                a["poseattr"]=[0]*len(include_poseattr)
            for i in a["poseattr"]:
                if i<0.0 or i>1.0:
                    print(f"Warning: poseattr {i} out of range")
                an_txt+=sstr(0)
                an_txt+=sstr(0)
                an_txt+=sstr(stuff.clip01(i))
        an_txt+="\n"
    return an_txt


def get_dataset_path(ds_yaml):
    return os.path.dirname(ds_yaml)

def get_dataset_name(ds_yaml):
    ds=stuff.load_dictionary(ds_yaml)
    if ds!=None:
        if "dataset_name" in ds:
            return ds["dataset_name"]
    return name_from_file(ds_yaml)

def get_dataset_config_name(ds_yaml):
    ds=stuff.load_dictionary(ds_yaml)
    if ds!=None:
        if "config_name" in ds:
            return ds["config_name"]
    if "dataset_name" in ds:
        config_name=ds["dataset_name"]
    else:
        config_name=name_from_file(ds_yaml)
    if "-" in config_name:
        config_name=config_name[0:config_name.find("-")]
    return config_name

def get_loader(name):
    loaders={"CocoLoader":CocoLoader,
             "OpenImagesLoader": OpenImagesLoader,
             "O365Loader": O365Loader,
             "WiderfaceLoader":WiderfaceLoader,
             "RoboflowLoader":RoboflowLoader,
             "WeaponDetectionLoader":WeaponDetectionLoader,
             "WiderAttributesLoader":WiderAttributesLoader}
    
    if not name in loaders:
        print("Could not find loader "+name)
        exit()
    return loaders[name]

def loader_name(loader):
    name=loader.lower()
    if "loader" in name:
        name=name[0:name.find("loader")]
    return name


llm_loaders={}

def register_llm_loader(name, f):
    print(f"Registering LLM loader {name}")
    llm_loaders[name]=f

def get_llm(name):
    name=name.lower()
    loaders={"openai": LLMOpenAI,
             "anthropic" : LLMAnthropic,
             "mobileclip" : LLMMobileCLIP,
             #"ollama" : LLMOllama
             }
    if not name in loaders:
        print("Could not find LLM "+name)
    return loaders[name]

def image_get_exit_data(image_file):
    comment=stuff.image_get_exif_comment(image_file)
    if comment is None:
        return None
    comment_lines=comment.split(";")
    ret={}
    items=["origin","config_name"]
    for c in comment_lines:
        for i in items:
            if i+"=" in c:
                ret[i]=c.split("=")[1]
    return ret

def box_to_coords(box, w, h):
    x0=int(box[0]*w)
    x1=int(box[2]*w)
    y0=int(box[1]*h)
    y1=int(box[3]*h)
    return x0,y0,x1,y1

def mask_detections(img, gts, dets, thr=0.5):
    h, w, c = img.shape
    orig = img.copy()
    for d in dets:
        if d["confidence"]>thr:
            x0,y0,x1,y1=box_to_coords(d["box"],w,h)
            img[y0:y1, x0:x1]=(80,80,80)
    for g in gts:
        x0,y0,x1,y1=box_to_coords(g["box"],w,h)
        img[y0:y1, x0:x1]=orig[y0:y1, x0:x1]

def facepose_iou(pose_points, pose_box, face_points, face_box):
    test_iou=stuff.box_iou(pose_box, face_box)
    if test_iou==0:
        return 0
    
    pose_xmin=1
    pose_xmax=0
    pose_ymin=1
    pose_ymax=0
    np=0
    for i in [0,1,2,5,6]:
        if pose_points[3*i+2]==0:
            continue
        pose_xmin=min(pose_points[3*i+0], pose_xmin)
        pose_xmax=max(pose_points[3*i+0], pose_xmax)
        pose_ymin=min(pose_points[3*i+1], pose_ymin)
        pose_ymax=max(pose_points[3*i+1], pose_ymax)
        np+=1
        if i==2 and np==3:
            break

    face_xmin=1
    face_xmax=0
    face_ymin=1
    face_ymax=0

    np=0
    for i in [0,1,2,3,4]:
        if face_points[3*i+2]==0:
            continue
        face_xmin=min(face_points[3*i+0], face_xmin)
        face_xmax=max(face_points[3*i+0], face_xmax)
        face_ymin=min(face_points[3*i+1], face_ymin)
        face_ymax=max(face_points[3*i+1], face_ymax)
        np+=1
        if i==2 and np==3:
            break

    face_point_box=[face_xmin, face_ymin, face_xmax, face_ymax]
    iou=stuff.box_iou([pose_xmin, pose_ymin, pose_xmax, pose_ymax], face_point_box)
    if iou>0:
        return iou
    ret=stuff.box_iou(pose_box, face_point_box)
    if ret==0:
        return test_iou/10.0
    return ret

def match_facepose(gts, person_class, face_class):
    person_indices=[]
    face_indices=[]
    for ii,g in enumerate(gts):
        if g["class"]==person_class:
            person_indices.append(ii)
        elif g["class"]==face_class:
            face_indices.append(ii)

    n_p=len(person_indices)
    n_f=len(face_indices)
    if n_p!=0 and n_f!=0:
        costs = [[0 for x in range(n_f)] for y in range(n_p)]
        for ii in range(n_p):
            for jj in range(n_f):
                costs[ii][jj]=facepose_iou(gts[person_indices[ii]]["pose_points"],
                                           gts[person_indices[ii]]["box"],
                                           gts[face_indices[jj]]["face_points"],
                                           gts[face_indices[jj]]["box"])
        row_ind, col_ind = linear_sum_assignment(np.array(costs), maximize=True)

        for k,_ in enumerate(row_ind):
            p_gt=gts[person_indices[row_ind[k]]]
            f_gt=gts[face_indices[col_ind[k]]]
            iou=stuff.box_iou(p_gt["box"],f_gt["box"])
            if iou==0:
                continue
            p_gt["face_points"]=copy.copy(f_gt["face_points"])
            map_one_gt_keypoints(p_gt, False, False, True)
            del f_gt["face_points"]
            del f_gt["pose_points"]

    for gt in gts:
        if gt["class"]==person_class:
            if not "facepose_points" in gt:
                map_one_gt_keypoints(gt, False, False, True)
            assert "facepose_points" in gt
            assert not "pose_points" in gt
        if gt["class"]==face_class:
            if "face_points" in gt or "pose_points" in gt:
                gt["class"]=person_class
                map_one_gt_keypoints(gt, False, False, True)
                assert not "pose_points" in gt
    for gt in gts:
        if "pose_points" in gt:
            del gt["pose_points"]

def facepose_facebox(gt):
    if "facepose_points" in gt:
        face_xmin=1
        face_xmax=0
        face_ymin=1
        face_ymax=0
        np=0
        facepose_points=gt["facepose_points"]
        conf=0
        for i in [0,1,2,17,18]:
            conf=max(conf, facepose_points[3*i+2])
            if facepose_points[3*i+2]==0:
                continue
            face_xmin=min(facepose_points[3*i+0], face_xmin)
            face_xmax=max(facepose_points[3*i+0], face_xmax)
            face_ymin=min(facepose_points[3*i+1], face_ymin)
            face_ymax=max(facepose_points[3*i+1], face_ymax)
        box=[face_xmin, face_ymin, face_xmax, face_ymax]
        if facepose_points[0*3+2]!=0:
            middle_x=facepose_points[0*3+0]
            middle_y=facepose_points[0*3+1]
        else:
            middle_x=(box[0]+box[2])*0.5
            middle_y=(box[1]+box[3])*0.5

        box[0]=stuff.clip01(middle_x-2.7*(middle_x-box[0]))
        box[2]=stuff.clip01(middle_x+2.7*(box[2]-middle_x))
        box[1]=stuff.clip01(middle_y-2.82*(middle_y-box[1]))
        box[3]=stuff.clip01(middle_y+2.42*(box[3]-middle_y))
        if box[2]<=box[0]:
            return None,0
        if box[3]<=box[1]:
            return None,0
        return box,conf
    return None,0

def add_faces_from_facepose(gts, person_class, face_class):
    faces=[]
    for g in gts:
        if g["class"]==person_class and "facepose_points" in g:
            f={}
            f["class"]=face_class
            f["confidence"]=1
            f["box"],_=facepose_facebox(g)
            f["facepose_points"]=copy.copy(g["facepose_points"])
            map_one_gt_keypoints(f, True, True, False)
            if f["box"]!=None:
                faces.append(f)
    gts+=faces

def print_gts(gts, classes=None):
    for g in gts:
        if g==None:
            continue
        if classes==None:
            s="{:10s}".format("class "+str(g["class"]))
        else:
            s="{:10s}".format(classes[g["class"]])
        s+=" C={:3.2f}".format(g["confidence"])
        s+=" B={:3.2f},{:3.2f}->{:3.2f},{:3.2f}".format(g["box"][0], g["box"][1], g["box"][2], g["box"][3])
        print(s)

def object_img(img, gt, scale_width=None, scale_height=None, expand=1.0, no_enlarge=False):
    height, width, _ = img.shape
    x0=width*gt["box"][0]
    x1=width*gt["box"][2]
    y0=height*gt["box"][1]
    y1=height*gt["box"][3]
    if expand!=1.0:
        w=x1-x0
        h=y1-y0
        w=0.5*w*(expand-1.0)
        h=0.5*h*(expand-1.0)
        x0-=w
        x1+=w
        y0-=h
        y1+=h
    x0=int(x0)
    x1=int(x1+0.99)
    y0=int(y0)
    y1=int(y1+0.99)
    x0=max(0,x0)
    y0=max(0,y0)
    x1=min(width,x1)
    y1=min(height,y1)
    person_img=img[y0:y1, x0:x1]

    if scale_width is not None or scale_height is not None:
        f=10
        if scale_width is not None:
            f=min(f, scale_width/(x1-x0))    
        if scale_height is not None:
            f=min(f, scale_height/(y1-y0))    
        if no_enlarge:
            f=min(f, 1.0)
        person_img=cv2.resize(person_img, None, fx=f, fy=f, interpolation = cv2.INTER_CUBIC)
    return person_img
        
def object_jpeg(img, gt, scale_width=None, scale_height=None, expand=1.0, no_enlarge=False):
    img=object_img(img, gt, scale_width, scale_height, expand, no_enlarge)
    success, encoded=cv2.imencode(".jpg", img)
    assert success
    return encoded

def base64_encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def get_param(task, config, param, default_value):
    if config is None:
        return default_value
    if param+"_val" in config and task=="val":
        return config[param+"_val"]
    if param+"_train" in config and task=="train":
        return config[param+"_train"]
    if param in config:
        return config[param]
    return default_value

def attributes_from_class_names(class_names):
    """
    Detect class names that correspond to attributes of a base
    class. Returns a list of attribtues
    e.g. person_male class -> person:male attribute
    """
    attributes=[]
    for c in class_names:
        if c.startswith("person_"):
            attributes.append(c.replace("person_", "person:"))
    return attributes

def timestr(delta):
    delta=int(delta)
    s=delta%60
    m=(delta//60)%60
    h=delta//3600
    r=f"{m:02d}:{s:02d}"
    if h!=0:
        r=f"{h:02d}:"+r
    return r

