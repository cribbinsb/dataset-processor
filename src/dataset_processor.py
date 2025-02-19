import src.dataset_util as dsu
import ultralytics
import cv2
import os
import random
import copy
import shutil
import pandas as pd
import torch
import concurrent.futures
import gc
import stuff

class DatasetProcessor:
    class_synonyms={'person':['man','woman','boy','girl'],
                    'vehicle':['car','bicycle','motorcycle','train','truck','bus','airplane','boat'],
                    'animal':['cat','dog','horse','sheep','cow','bird','elephant','bear','zebra','giraffe'],
                    'weapon':['gun', 'dagger', 'pistol', 'handgun', 'rifle', 'revolver'],
                    'face':[]}
    
    def __init__(self, ds_yaml, task="val",
                 class_names=None,
                 chunk_size=0,
                 append_log=False, print_log=False, face_kp=True, pose_kp=True,
                 facepose_kp=False, fold_attributes=False,
                 poseattr=None):
        self.chunk_size=chunk_size
        self.num_chunks=0
        self.ds_yaml=ds_yaml
        self.task=task
        self.reload_files()
        self.config_name=dsu.get_dataset_config_name(ds_yaml)
        assert self.ds_files is not None
        self.face_kp=face_kp
        self.pose_kp=pose_kp
        self.facepose_kp=facepose_kp
        self.poseattr=poseattr
        self.dataset_path=os.path.dirname(ds_yaml)
        self.append_log=append_log
        self.print_log=print_log
        self.fold_attributes=False
        if class_names is None:
            class_names=self.ds_class_names
            if fold_attributes and self.attributes is None:
                self.attributes=dsu.attributes_from_class_names(self.ds_class_names)
                print(f"Folded classes->attributes {self.attributes}")
                self.fold_attributes=True
        self.class_names=class_names
        self.yolo=None
        self.yolo_nms_iou=0.6
        self.yolo_max_det=600
        self.yolo_det_conf=0.001

    def reload_files(self):
        self.ds_files, self.ds_class_names, self.dataset_name, self.attributes=dsu.get_all_dataset_files(self.ds_yaml, self.task, "image")
        self.num_files=len(self.ds_files)
        if self.chunk_size!=0:
            self.num_chunks=(self.num_files+self.chunk_size-1)//self.chunk_size
            self.current_chunk=0
            self.ds_files_all=self.ds_files
            self.num_files_all=self.num_files
            self.set_chunk(0)

    def __len__(self):
        return len(self.ds_files)

    def __getitem__(self, index):
        return self.ds_files[index]

    def log(self, txt):
        if self.print_log:
            print("log: "+str(txt))
        if self.append_log:
            dsu.append_comments(self.ds_yaml, txt)

    def basic_stats(self):
        class_counts=[0]*len(self.class_names)
        image_count=0
        background_count=0
        face_point_count=0
        pose_point_count=0
        small_persons_count=0
        attributes_count=0
        person_class=self.get_class_index("person")
        poseattr_count=None
        if self.poseattr is not None:
            poseattr_count=[0]*len(self.poseattr)
        for i in range(self.num_files):
            gts=self.get_gt(i)
            has_small_persons=False
            for g in gts:
                class_counts[g["class"]]+=1
                if dsu.has_face_points(g):
                    face_point_count+=1
                if dsu.has_pose_points(g):
                    pose_point_count+=1
                if g["class"]==person_class:
                    if not dsu.is_large(g):
                        has_small_persons=True
                if "attrs" in g:
                    attributes_count+=1
                if "poseattr" in g:
                    for k in range(len(g["poseattr"])):
                        poseattr_count[k]+=g["poseattr"][k]
            if has_small_persons:
                small_persons_count+=1
            if len(gts)==0:
                background_count+=1
            image_count+=1

        ret={"task":self.task, 
            "class_counts":class_counts, 
            "image_count":image_count, 
            "backgrounds":background_count, 
            "image_small_person_count":small_persons_count, 
            "has_face_kp":face_point_count, 
            "has_pose_kp":pose_point_count,
            "has_attributes":attributes_count}
        if poseattr_count is not None:
            ret["poseattr"]=[int(k) for k in poseattr_count]
        return ret

    def log_stats(self):
        stats=self.basic_stats()
        self.log(str(stats))

    def set_chunk(self, cn):
        self.current_chunk=cn
        start=cn*self.chunk_size
        end=min((cn+1)*self.chunk_size, self.num_files_all)
        self.num_files=end-start
        self.ds_files=self.ds_files_all[start:end]

    def get_gt(self, index):
        if index>=len(self.ds_files):
            print(f"ERROR: get_gt index {index} not in image list {len(self.ds_files)}")
            return None
        label_file=self.ds_files[index]["label"]
        num_poseattr=None
        if self.poseattr is not None:
            num_poseattr=len(self.poseattr)
        gts=dsu.load_ground_truth_labels(label_file, num_poseattr=num_poseattr)
        if gts is None:
            return None
        gt_class_remap=[self.class_names.index(x) if x in self.class_names else -1 for x in self.ds_class_names]
        for g in gts:
            ci=g["class"]
            if ci>=len(gt_class_remap) or ci<0:
                print(f"ERROR: bad GT class index {ci} (max {len(gt_class_remap)})")
            g["class"]=gt_class_remap[ci]
        ret=[g for g in gts if g["class"]!=-1]
        if self.fold_attributes:
            ret=stuff.fold_detections_to_attributes(ret, self.class_names, self.attributes)
        return ret
    
    def get_gt_file_path(self, index):
        label_file=self.ds_files[index]["label"]
        return label_file
    
    def get_class_names(self):
        return self.class_names
    
    def get_class_index(self, class_name):
        if class_name in self.class_names:
            return self.class_names.index(class_name)
        return -1
    
    def add(self, name, img_file, dets, add_exif=False):
        stuff.map_keypoints(dets, self.face_kp, self.pose_kp, self.facepose_kp)
        dst_img=self.dataset_path+"/"+self.task+"/images/"+name+".jpg"
        dst_label=self.dataset_path+"/"+self.task+"/labels/"+name+".txt"
        shutil.copyfile(img_file, dst_img)
        if add_exif:
            ok=stuff.image_append_exif_comment(dst_img, f"config={self.config_name};origin={img_file}")
            if ok is False:
                print(f"dataset_processor add: rejecting file {img_file}")
                stuff.rm(dst_img)
                return False
        an_txt=dsu.write_annotations(dets, 
                                     include_face=self.face_kp, 
                                     include_pose=self.pose_kp, 
                                     include_facepose=self.facepose_kp, 
                                     include_poseattr=self.poseattr,
                                     include_attrs=True)
        with open(dst_label, 'w') as file:
            file.write(an_txt)
        return True

    def delete(self, index):
        stuff.rm(self.ds_files[index]["label"])
        stuff.rm(self.ds_files[index]["image"])
        self.ds_files[index]["label"]=None
        self.ds_files[index]["image"]=None

    def rename(self, index, name):
        dst_img=self.dataset_path+"/"+self.task+"/images/"+name+".jpg"
        dst_label=self.dataset_path+"/"+self.task+"/labels/"+name+".txt"
        stuff.rename(self.ds_files[index]["label"], dst_label)
        stuff.rename(self.ds_files[index]["image"], dst_img)
        self.ds_files[index]["label"]=dst_label
        self.ds_files[index]["image"]=dst_img

    def get_exif_data(self, index):
        return dsu.image_get_exit_data(self.ds_files[index]["image"])

    def append_exif_comment(self, index, comment):  
        stuff.image_append_exif_comment(self.ds_files[index]["image"], comment)
    
    def load_ultralytics_model(self, name):
        task="detect"
        if "world" in name:
            extended_classes=copy.deepcopy(self.class_names)
            for c in self.class_names:
                if c in self.class_synonyms:
                    for s in self.class_synonyms[c]:
                        if not s in extended_classes:
                            extended_classes.append(s)

            model=ultralytics.YOLOWorld(name, verbose=False)
            model.set_classes(extended_classes)
            return model
        if "nas" in name:
            return ultralytics.NAS(name)
        if "rtdetr" in name:
            return ultralytics.RTDETR(name)
        if "pose" in name or "face" in name or "full" in name:
            task="pose"
        return ultralytics.YOLO(name, task=task, verbose=False)

    def set_yolo_detector(self, yolo,
                          imgsz=640,
                          thr=0.001,
                          rect=False,
                          half=True,
                          batch_size=32,
                          augment=False,
                          remap_pose=True):

        if self.yolo!=None:
            del self.yolo
            self.yolo=None

        gc.collect()
        torch.cuda.empty_cache() # try hard not to run out of GPU memory
        self.yolo_cache={}

        if yolo==None:
            return

        self.imgsz=imgsz
        self.yolo_batch_size=batch_size
        self.yolo_det_conf=max(thr, 0.001)
        self.yolo_rect=rect
        self.yolo_half=half
        self.num_gpus=torch.cuda.device_count()
        assert(self.num_gpus>0)
        self.yolo_remap_pose=remap_pose
        self.yolo_add_faces=True

        if type(yolo) is str:
            if ":" in yolo:
                t=yolo.split(":")
                yolo=t[0]
                self.imgsz=int(t[1])
                self.yolo_batch_size=self.yolo_batch_size//2

            if self.num_gpus==1:
                self.yolo=self.load_ultralytics_model(yolo)
                assert self.yolo!=None
                self.yolo_class_names=[self.yolo.names[i] for i in range(len(self.yolo.names))]
            else:
                self.yolo=[None]*self.num_gpus
                assert self.yolo!=None
                for i in range(self.num_gpus):
                    self.yolo[i]=self.load_ultralytics_model(yolo)
                    self.yolo[i]=self.yolo[i].to("cuda:"+str(i))
                    assert(self.yolo[i] is not None)
                self.yolo_class_names=[self.yolo[0].names[i] for i in range(len(self.yolo[0].names))]
        elif type(yolo) is list:
            self.yolo=yolo
            self.yolo_class_names=[self.yolo[0].names[i] for i in range(len(self.yolo[0].names))]
        else:
            self.yolo=yolo
            self.yolo_class_names=[self.yolo.names[i] for i in range(len(self.yolo.names))]
        self.yolo_cache={}

        try:
            if type(self.yolo) is list:
                self.yolo_num_params = sum(p.numel() for p in self.yolo[0].model.parameters())
            else:
                self.yolo_num_params = sum(p.numel() for p in self.yolo.model.parameters())
        except AttributeError:
            self.yolo_num_params = 0

        # map the detected classes back to our class set
        # assume if our class set has things like 'vehicle' then we would want any standard
        # coco classes like 'car' to map to that

        self.det_class_remap=[-1]*len(self.yolo_class_names)
        self.can_detect=[False]*len(self.class_names)
        for i,x in enumerate(self.yolo_class_names):
            if x in self.class_names:
                self.det_class_remap[i]=self.class_names.index(x)
                self.can_detect[self.class_names.index(x)]=True
            else:
                for y in self.class_synonyms:
                    if y in self.class_names:
                        if x in self.class_synonyms[y]:
                            self.det_class_remap[i]=self.class_names.index(y) 
                            self.can_detect[self.class_names.index(y)]=True  

    def get_img(self, index):
        return cv2.imread(self.ds_files[index]["image"])
    
    def replace_img(self, index, image):
        cv2.imwrite(self.ds_files[index]["image"], image)
    
    def get_img_path(self, index):
        return self.ds_files[index]["image"]
    
    def get_label_path(self, index):
        return self.ds_files[index]["label"]
    
    def get_img_name(self, index):
        label_file=self.ds_files[index]["label"]
        return dsu.name_from_file(label_file)

    def replace_annotations(self, index, an):
        txt=dsu.write_annotations(an, include_face=self.face_kp, include_pose=self.pose_kp, include_facepose=self.facepose_kp, include_attrs=True)
        label_file=self.get_gt_file_path(index)
        with open(label_file, 'w') as file:
            file.write(txt)

    def get_detections(self, index, det_thr=0.01):
        out_det=[]
        if self.yolo!=None:
            if not index in self.yolo_cache:
                if self.num_gpus==1:
                    input_frames=[self.get_img(x) for x in range(index, min(self.num_files, index+self.yolo_batch_size))]
                    #print(f"run yolo on {input_frames}")
                    batch_result=self.yolo(input_frames, conf=self.yolo_det_conf, iou=self.yolo_nms_iou, max_det=self.yolo_max_det, agnostic_nms=False, half=self.yolo_half, imgsz=self.imgsz, verbose=False, rect=self.yolo_rect)
                    self.yolo_cache={(i+index):batch_result[i] for i in range(len(input_frames))}
                else:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        thread_state=[]
                        for i in range(self.num_gpus):
                            start=min(self.num_files, index+i*self.yolo_batch_size)
                            end=min(self.num_files, index+(i+1)*self.yolo_batch_size)
                            if start!=end:
                                input_frames=[self.get_img(x) for x in range(start, end)]
                                future = executor.submit(self.yolo[i], input_frames, conf=self.yolo_det_conf, iou=self.yolo_nms_iou, max_det=self.yolo_max_det, agnostic_nms=False, half=self.yolo_half, imgsz=self.imgsz, verbose=False, rect=self.yolo_rect)
                                thread_state.append({"frames":input_frames, "future":future, "start":start})
                        self.yolo_cache={}
                        for i,_ in enumerate(thread_state):
                            batch_result=thread_state[i]["future"].result()
                            input_frames=thread_state[i]["frames"]
                            
                            for j in range(len(input_frames)):
                                self.yolo_cache[j+thread_state[i]["start"]]=batch_result[j]

            assert(index in self.yolo_cache)
            
            results=self.yolo_cache[index]
            
            out_det=stuff.yolo_results_to_dets(results,
                                             det_thr=det_thr,
                                             det_class_remap=self.det_class_remap,
                                             yolo_class_names=self.yolo_class_names,
                                             class_names=self.class_names,
                                             attributes=self.attributes,
                                             add_faces=self.yolo_add_faces,
                                             face_kp=self.face_kp,
                                             pose_kp=self.pose_kp,
                                             facepose_kp=self.facepose_kp,
                                             fold_attributes=self.fold_attributes)
        return out_det

def dataset_merge_and_delete(src, dest):
    dsu.append_comments(dest, "====== Merge dataset "+src+" ======")
    dsu.dataset_copy_comments(src, dest)

    dest_path=dsu.get_dataset_path(dest)

    for task in ["val","train"]:
        x=DatasetProcessor(src, task=task)
        for i in range(x.num_files):    
            img=x.get_img_path(i)
            label=x.get_label_path(i)
            stuff.rename(img, dest_path+"/"+task+"/images/"+os.path.basename(img))
            stuff.rename(label, dest_path+"/"+task+"/labels/"+os.path.basename(label))

    stuff.rmdir(os.path.dirname(src))

def dataset_delete(dataset_yaml):
    stuff.rmdir(os.path.dirname(src))
