import os, glob, sys
import logging

import torch
import torch.nn.functional as torchfn
from torchvision.transforms.functional import normalize
from torchvision.ops import masks_to_boxes

import numpy as np
import cv2
import math
from typing import List
from PIL import Image
from scipy import stats
from insightface.app.common import Face
from segment_anything import sam_model_registry

from modules.processing import StableDiffusionProcessingImg2Img
from modules.shared import state
# from comfy_extras.chainner_models import model_loading
import comfy.model_management as model_management
import comfy.utils
import folder_paths

import scripts.reactor_version
from r_chainner import model_loading
from scripts.reactor_faceswap import (
    FaceSwapScript,
    get_models,
    get_current_faces_model,
    analyze_faces,
    half_det_size,
    providers
)
from scripts.reactor_logger import logger
from reactor_utils import (
    batch_tensor_to_pil,
    batched_pil_to_tensor,
    tensor_to_pil,
    img2tensor,
    tensor2img,
    save_face_model,
    load_face_model,
    download,
    set_ort_session,
    prepare_cropped_face,
    normalize_cropped_face,
    add_folder_path_and_extensions,
    rgba2rgb_tensor
)
from reactor_patcher import apply_patch
from r_facelib.utils.face_restoration_helper import FaceRestoreHelper
from r_basicsr.utils.registry import ARCH_REGISTRY
import scripts.r_archs.codeformer_arch
import scripts.r_masking.subcore as subcore
import scripts.r_masking.core as core
import scripts.r_masking.segs as masking_segs
from ultralytics import YOLO
from scripts.reactor_swapper import (
    getFaceSwapModel, 
    insightface_path
)


models_dir = folder_paths.models_dir
REACTOR_MODELS_PATH = os.path.join(models_dir, "reactor")
FACE_MODELS_PATH = os.path.join(REACTOR_MODELS_PATH, "faces")

if not os.path.exists(REACTOR_MODELS_PATH):
    os.makedirs(REACTOR_MODELS_PATH)
    if not os.path.exists(FACE_MODELS_PATH):
        os.makedirs(FACE_MODELS_PATH)

dir_facerestore_models = os.path.join(models_dir, "facerestore_models")
os.makedirs(dir_facerestore_models, exist_ok=True)
folder_paths.folder_names_and_paths["facerestore_models"] = ([dir_facerestore_models], folder_paths.supported_pt_extensions)

BLENDED_FACE_MODEL = None
FACE_SIZE: int = 512
FACE_HELPER = None

if "ultralytics" not in folder_paths.folder_names_and_paths:
    add_folder_path_and_extensions("ultralytics_bbox", [os.path.join(models_dir, "ultralytics", "bbox")], folder_paths.supported_pt_extensions)
    add_folder_path_and_extensions("ultralytics_segm", [os.path.join(models_dir, "ultralytics", "segm")], folder_paths.supported_pt_extensions)
    add_folder_path_and_extensions("ultralytics", [os.path.join(models_dir, "ultralytics")], folder_paths.supported_pt_extensions)
if "sams" not in folder_paths.folder_names_and_paths:
    add_folder_path_and_extensions("sams", [os.path.join(models_dir, "sams")], folder_paths.supported_pt_extensions)


def get_restorers():
    models_path = os.path.join(models_dir, "facerestore_models/*")
    models = glob.glob(models_path)
    models = [x for x in models if (x.endswith(".pth") or x.endswith(".onnx"))]
    if len(models) == 0:
        fr_urls = [
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.3.pth",
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.pth",
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/codeformer-v0.1.0.pth",
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-512.onnx",
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-1024.onnx",
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-2048.onnx",
        ]
        for model_url in fr_urls:
            model_name = os.path.basename(model_url)
            model_path = os.path.join(dir_facerestore_models, model_name)
            download(model_url, model_path, model_name)
        models = glob.glob(models_path)
        models = [x for x in models if (x.endswith(".pth") or x.endswith(".onnx"))]
    return models

def get_model_names(get_models):
    models = get_models()
    names = []
    for x in models:
        names.append(os.path.basename(x))
    names.sort(key=str.lower)
    names.insert(0, "none")
    return names

def model_names():
    models = get_models()
    return {os.path.basename(x): x for x in models}


DIR_REACTOR_V2_MODELS = os.path.join(models_dir, "reactor_v2")
os.makedirs(DIR_REACTOR_V2_MODELS, exist_ok=True)

FACE_DETECTION_MODELS_PATH = os.path.join(DIR_REACTOR_V2_MODELS, "face_detection_models")
os.makedirs(FACE_DETECTION_MODELS_PATH, exist_ok=True)

def get_face_detection_models():
    models_path = os.path.join(FACE_DETECTION_MODELS_PATH, "*")
    models = glob.glob(models_path)
    models = [x for x in models if (x.endswith(".pth") or x.endswith(".onnx") or x.endswith(".pt"))]
    if len(models) == 0:
        fr_urls = [
            "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11n-face.pt",
            "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11s-face.pt",
            "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11m-face.pt",
            # "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov10n-face.pt",
            # "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov10s-face.pt",
            # "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov10m-face.pt",
            # "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt",
            # "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8m-face.pt",
            # "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8l-face.pt"
        ]
        for model_url in fr_urls:
            model_name = os.path.basename(model_url)
            model_path = os.path.join(FACE_DETECTION_MODELS_PATH, model_name)
            download(model_url, model_path, model_name)
        models = glob.glob(models_path)
        models = [x for x in models if (x.endswith(".pth") or x.endswith(".onnx") or x.endswith(".pt"))]
    return models

class FaceDetectionModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_detection_model":(get_model_names(get_face_detection_models), ),
            }
        }
    RETURN_TYPES = ("FACE_DETECTION_MODEL",)
    FUNCTION = "load_face_detection_model"
    CATEGORY = "🌌 ReActor_v2"
    
    def load_face_detection_model(self, face_detection_model):
        self.face_detection_model = face_detection_model
        self.face_detection_models_path = FACE_DETECTION_MODELS_PATH
        if self.face_detection_model != "none":
            face_model_path = os.path.join(self.face_detection_models_path, self.face_detection_model)
            model = YOLO(face_model_path)
        else:
            model = None
        return (model, )

class FaceFeatureExtractor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "compute_method": (["Mean", "Median", "Mode"], {"default": "Mean"}),
            },
            "optional": {
                "images":("IMAGE", ),
                "face_detection_model":("FACE_DETECTION_MODEL", ),
            }
        }
    RETURN_TYPES = ("FACE_FEATURE", )
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor_v2"
    
    def extract_face_feature(self, image: Image.Image, det_size=(640, 640)):
        logging.StreamHandler.terminator = "\n"
        if image is None:
            error_msg = "Please load an Image"
            logger.error(error_msg)
            return error_msg
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces = analyze_faces(image, det_size)
        
        if len(faces) == 0:
            print("")
            det_size_half = half_det_size(det_size)
            faces = analyze_faces(image, det_size_half)
            if faces is not None and len(faces) > 0:
                print("...........................................................", end=" ")
        
        if faces is not None and len(faces) > 0:
            # print("face_model=",face_model)
            return faces[0]
        else:
            no_face_msg = "No face found, please try another image"
            # logger.error(no_face_msg)
            return no_face_msg
        

    def execute(self, compute_method, images=None, face_detection_model=None):
        if images is None or face_detection_model is None:
            logger.error("Please provide `images` or `face_detection_model`")
            return (None,)
        
        faces = []
        embeddings=[]
        images_list: List[Image.Image] = batch_tensor_to_pil(images)
        for i, image in enumerate(images_list):
            face = self.extract_face_feature(image)
            faces.append(face)
            embeddings.append(face.embedding)
            
        logging.StreamHandler.terminator = "\n"
        if len(faces) > 0:
            logger.status(f"Blending with Compute Method '{compute_method}'...")
            blended_embedding = np.mean(embeddings, axis=0) if compute_method == "Mean" else np.median(embeddings, axis=0) if compute_method == "Median" else stats.mode(embeddings, axis=0)[0].astype(np.float32)
            logger.status("--Done!--")
        return (blended_embedding,)


class FaceSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "swap_model": (list(model_names().keys()),),
                "face_restore_model": (get_model_names(get_restorers),),
                "face_feature_update": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "similarity_threshold":("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "face_feature": ("FACE_FEATURE", ),
                "source_image": ("IMAGE", ),
                "face_detection_model":("FACE_DETECTION_MODEL", ),
            }
        }
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor_v2"
    
    def __init__(self):
        self.similarity_threshold = 0.5
        self.face_feature_update = True
        self.face_features = []
        
    def execute(self, input_image, swap_model, face_restore_model, face_feature_update, similarity_threshold=0.5, face_feature=None, source_image=None, face_detection_model=None):
        self.similarity_threshold = similarity_threshold
        self.face_feature_update = face_feature_update
        
        if face_feature is None:
            logger.info("Please provide `face_feature`")
            return (input_image,)
        if source_image is None:
            logger.info("Please provide `source_image`")
            return (input_image,)
        if face_detection_model is None:
            logger.info("Please provide `face_detection_model`")
            return (input_image,)
            
        self.face_features.append(face_feature)
        
        pil_images = batch_tensor_to_pil(input_image)
        if source_image is not None:
            source_img = tensor_to_pil(source_image)
        else:
            source_img = None
        
        if swap_model is not None:
            model_path = model_path = os.path.join(insightface_path, swap_model)
            face_swapper = getFaceSwapModel(model_path)
            
            if isinstance(source_img, str):  # source_img is a base64 string
                import base64, io
                if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                    # split the base64 string to get the actual base64 encoded image data
                    base64_data = source_img.split('base64,')[-1]
                    # decode base64 string to bytes
                    img_bytes = base64.b64decode(base64_data)
                else:
                    # if no data URL scheme, just decode
                    img_bytes = base64.b64decode(source_img)
                
                source_img = Image.open(io.BytesIO(img_bytes))
            
            # if len(pil_images) == 1:
            #     target_imgs = [cv2.cvtColor(np.array(pil_images), cv2.COLOR_RGB2BGR)]
            # else:
            target_imgs = [cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR) for target_img in pil_images]
            
            if source_img is not None:
                source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
            
            source_faces = analyze_faces(source_img)
            source_face = source_faces[0]
            
            def calc_similarity(vec1, vec2):
                dot_product = np.dot(vec1, vec2)
                norm_vec1 = np.linalg.norm(vec1)
                norm_vec2 = np.linalg.norm(vec2)
                return dot_product / (norm_vec1 * norm_vec2)
            
            results = target_imgs
            for i, target_img in enumerate(target_imgs):
                result = target_img
                
                target_faces = analyze_faces(target_img)
                similarities=[]
                for tface in target_faces:
                    target_similarities=[]
                    for feature in self.face_features:
                        similarity = calc_similarity(tface.embedding, feature)
                        target_similarities.append(similarity)
                    max_similarity = max(target_similarities)
                    similarities.append(max_similarity)

                target_face_single = None
                if max(similarities) >= self.similarity_threshold:
                    tindex = np.argmax(similarities)
                    target_face_single = target_faces[tindex]
                    if self.face_feature_update:
                        self.face_features.append(target_face_single.embedding)

                if target_face_single is not None:
                    result = face_swapper.get(target_img, target_face_single, source_face)
                    # bgr_fake, M = face_swapper.get(target_img, target_face_single, source_face, paste_back=False)
                    # bgr_fake, scale = restorer.get_restored_face(bgr_fake, face_restore_model, face_restore_visibility, codeformer_weight, interpolation)
                    # M *= scale
                    # result = swapper.in_swap(target_img, bgr_fake, M)
                
                results[i] = result

            result_images = [Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) for result in results]
            
            result_images = batched_pil_to_tensor(result_images)
        return (result_images, )


NODE_CLASS_MAPPINGS = {
    # --- MAIN NODES ---
    "ReActorFaceSwap_v2": FaceSwap,
    # --- Additional Nodes ---
    
    # Mine
    "FaceDetectionModelLoader": FaceDetectionModelLoader,
    "FaceFeatureExtractor": FaceFeatureExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # --- MAIN NODES ---
    "ReActorFaceSwap_v2": "ReActor_v2 🌌 Face Swap(Embedding)",
    # --- Operations with Face Models ---
    "FaceDetectionModelLoader": "ReActor_v2 🌌 FaceDetectionModelLoader",
    "FaceFeatureExtractor": "ReActor_v2 🌌 FaceFeatureExtractor",
}
