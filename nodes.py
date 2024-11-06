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
from scripts.r_faceboost import swapper, restorer


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


class FaceSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "target_images": ("IMAGE",),
                "swap_model": (list(model_names().keys()),),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "face_restore_model": (get_model_names(get_restorers),),
                "face_restore_visibility": ("FLOAT", {"default": 1, "min": 0.1, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
                "face_feature_update": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "similarity_threshold":("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "source_images": ("IMAGE", ),
            }
        }
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor_v2"
    
    def __init__(self):
        self.similarity_threshold = 0.5
        self.face_feature_update = True
        self.face_features = []
        self.face_boost_enabled = False
        self.restore = True
        self.boost_model = None
        self.interpolation = "Bicubic"
        self.boost_model_visibility = 1
        self.boost_cf_weight = 0.5


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
        
    def extract_feature(self, compute_method="Mean", image: Image.Image=None):
        if image is None:
            logger.error("Please provide `target_images`")
            return None
        # pil_image: Image.Image = tensor_to_pil(image)
        face = self.extract_face_feature(image)
        return [face.embedding]
                    
    def execute(self, 
                input_image, 
                target_images, 
                swap_model, 
                facedetection, 
                face_restore_model, 
                face_restore_visibility, 
                codeformer_weight, 
                face_feature_update, 
                similarity_threshold=0.5, 
                source_images=None):
        
        self.similarity_threshold = similarity_threshold
        self.face_feature_update = face_feature_update

        # print('target_images=', target_images)
        target_pil_images = batch_tensor_to_pil(target_images)
        for i, target_pil_image in enumerate(target_pil_images):
            face_feature = self.extract_feature(image=target_pil_image)
            self.face_features.append(face_feature)
        print('extract features in target_images=', len(self.face_features))
        
        pil_images = batch_tensor_to_pil(input_image)

        if source_images is None:
            logger.info("Please provide `source_images`")
            return (input_image,)
        else:
            source_pil_imgs = batch_tensor_to_pil(source_images)
        # else:
        #     source_pil_imgs = None
        
        if swap_model is not None:
            model_path = model_path = os.path.join(insightface_path, swap_model)
            face_swapper = getFaceSwapModel(model_path)
            
            target_imgs = [cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR) for target_img in pil_images]
            
            if source_pil_imgs is not None:
                source_imgs = [cv2.cvtColor(np.array(source_pil_img), cv2.COLOR_RGB2BGR) for source_pil_img in source_pil_imgs]
            
            source_faces=[]
            for i, source_img in enumerate(source_imgs):
                # print("source_img=",source_img)
                faces = analyze_faces(source_img)
                # print(f"{i}-", faces)
                source_faces.append(faces[0])
            
            def similarity(vec1, vec2):
                dot_product = np.dot(vec1, vec2)
                norm_vec1 = np.linalg.norm(vec1)
                norm_vec2 = np.linalg.norm(vec2)
                return dot_product / (norm_vec1 * norm_vec2)
            
            def calc_similarity(vec:np.array, veclist:List[np.array]):
                sims=[]
                for vec2 in veclist:
                    sim = similarity(vec, vec2)
                    sims.append(sim)
                return max(sims)
            
            results = target_imgs
            for i, target_img in enumerate(target_imgs):
                result = target_img
                
                target_faces = analyze_faces(target_img)
                # similarities=[]
                for tface in target_faces:
                    similarities=[]
                    for features in self.face_features:
                        sim = calc_similarity(tface.embedding, features)
                        similarities.append(sim)
                    max_similarity = max(similarities)
                    # similarities.append(max_similarity)
                    if max_similarity >= self.similarity_threshold:
                        sindex = np.argmax(similarities)
                        source_face = source_faces[sindex]
                        self.face_features[sindex].append(tface.embedding)
                        
                        if self.face_boost_enabled:
                            logger.status(f"Face Boost is enabled")
                            result = face_swapper.get(result, tface, source_face)
                            bgr_fake, M = face_swapper.get(result, tface, source_face, paste_back=False)
                            bgr_fake, scale = restorer.get_restored_face(bgr_fake, face_restore_model, face_restore_visibility, codeformer_weight, self.interpolation)
                            M *= scale
                            result = swapper.in_swap(result, bgr_fake, M)
                        else:
                            result = face_swapper.get(result, tface, source_face)
                results[i] = result

            result_images = [Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) for result in results]
            result_images = batched_pil_to_tensor(result_images)

            if self.restore:
                result_images = FaceSwap.restore_face(self,result_images,face_restore_model,face_restore_visibility,codeformer_weight,facedetection)
            
        return (result_images, )


    def restore_face(
            self,
            input_image,
            face_restore_model,
            face_restore_visibility,
            codeformer_weight,
            facedetection,
        ):

        result = input_image

        if face_restore_model != "none" and not model_management.processing_interrupted():

            global FACE_SIZE, FACE_HELPER

            self.face_helper = FACE_HELPER
            
            faceSize = 512
            if "1024" in face_restore_model.lower():
                faceSize = 1024
            elif "2048" in face_restore_model.lower():
                faceSize = 2048

            logger.status(f"Restoring with {face_restore_model} | Face Size is set to {faceSize}")

            model_path = folder_paths.get_full_path("facerestore_models", face_restore_model)

            device = model_management.get_torch_device()

            if "codeformer" in face_restore_model.lower():

                codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
                    dim_embd=512,
                    codebook_size=1024,
                    n_head=8,
                    n_layers=9,
                    connect_list=["32", "64", "128", "256"],
                ).to(device)
                checkpoint = torch.load(model_path)["params_ema"]
                codeformer_net.load_state_dict(checkpoint)
                facerestore_model = codeformer_net.eval()

            elif ".onnx" in face_restore_model:

                ort_session = set_ort_session(model_path, providers=providers)
                ort_session_inputs = {}
                facerestore_model = ort_session

            else:

                sd = comfy.utils.load_torch_file(model_path, safe_load=True)
                facerestore_model = model_loading.load_state_dict(sd).eval()
                facerestore_model.to(device)
            
            if faceSize != FACE_SIZE or self.face_helper is None:
                self.face_helper = FaceRestoreHelper(1, face_size=faceSize, crop_ratio=(1, 1), det_model=facedetection, save_ext='png', use_parse=True, device=device)
                FACE_SIZE = faceSize
                FACE_HELPER = self.face_helper

            image_np = 255. * result.numpy()

            total_images = image_np.shape[0]

            out_images = []

            for i in range(total_images):

                if total_images > 1:
                    logger.status(f"Restoring {i+1}")

                cur_image_np = image_np[i,:, :, ::-1]

                original_resolution = cur_image_np.shape[0:2]

                if facerestore_model is None or self.face_helper is None:
                    return result

                self.face_helper.clean_all()
                self.face_helper.read_image(cur_image_np)
                self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                self.face_helper.align_warp_face()

                restored_face = None

                for idx, cropped_face in enumerate(self.face_helper.cropped_faces):

                    # if ".pth" in face_restore_model:
                    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                    try:

                        with torch.no_grad():

                            if ".onnx" in face_restore_model: # ONNX models

                                for ort_session_input in ort_session.get_inputs():
                                    if ort_session_input.name == "input":
                                        cropped_face_prep = prepare_cropped_face(cropped_face)
                                        ort_session_inputs[ort_session_input.name] = cropped_face_prep
                                    if ort_session_input.name == "weight":
                                        weight = np.array([ 1 ], dtype = np.double)
                                        ort_session_inputs[ort_session_input.name] = weight

                                output = ort_session.run(None, ort_session_inputs)[0][0]
                                restored_face = normalize_cropped_face(output)

                            else: # PTH models

                                output = facerestore_model(cropped_face_t, w=codeformer_weight)[0] if "codeformer" in face_restore_model.lower() else facerestore_model(cropped_face_t)[0]
                                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

                        del output
                        torch.cuda.empty_cache()

                    except Exception as error:

                        print(f"\tFailed inference: {error}", file=sys.stderr)
                        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                    if face_restore_visibility < 1:
                        restored_face = cropped_face * (1 - face_restore_visibility) + restored_face * face_restore_visibility

                    restored_face = restored_face.astype("uint8")
                    self.face_helper.add_restored_face(restored_face)

                self.face_helper.get_inverse_affine(None)

                restored_img = self.face_helper.paste_faces_to_input_image()
                restored_img = restored_img[:, :, ::-1]

                if original_resolution != restored_img.shape[0:2]:
                    restored_img = cv2.resize(restored_img, (0, 0), fx=original_resolution[1]/restored_img.shape[1], fy=original_resolution[0]/restored_img.shape[0], interpolation=cv2.INTER_AREA)

                self.face_helper.clean_all()

                # out_images[i] = restored_img
                out_images.append(restored_img)

                if state.interrupted or model_management.processing_interrupted():
                    logger.status("Interrupted by User")
                    return input_image

            restored_img_np = np.array(out_images).astype(np.float32) / 255.0
            restored_img_tensor = torch.from_numpy(restored_img_np)

            result = restored_img_tensor

        return result
    

NODE_CLASS_MAPPINGS = {
    # --- MAIN NODES ---
    "ReActorFaceSwap_v2": FaceSwap,
    # --- Additional Nodes ---
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # --- MAIN NODES ---
    "ReActorFaceSwap_v2": "ReActor_v2 🌌 Face Swap(Embedding)",
    # --- Additional Nodes ---

}
