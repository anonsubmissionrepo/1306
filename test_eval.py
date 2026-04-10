import argparse
import os
import torch
import torch.nn.functional as F
import imageio
import numpy as np
from tqdm import tqdm
import time
import copy

from ptflops import get_model_complexity_info
from py_sod_metrics import Smeasure, Emeasure, MAE, WeightedFmeasure
import py_sod_metrics

from LoRA_SAM3 import LoRA_SAM3
import dataset as TestDataset 

from helpers.benchmark import print_trainable_params
from helpers.load_sam3 import load_sam3, create_vit_backbone, load_sam3_checkpoint_to_504

binarize = False
THRESHOLD = 0.5

evaluate = True

parser = argparse.ArgumentParser("SAM3DUNet_V4 Evaluation")
parser.add_argument("--save_images", type=bool, default=False)
parser.add_argument("--size", type=int, default=672)
parser.add_argument("--sam3_path", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--path", type=str, default="./test_data/")
parser.add_argument("--save_path", type=str, default="./preds/")
args = parser.parse_args()


def load_model(sam3_path, checkpoint_path, device):
    # Load SAM3 encoder
    sam3_vit = create_vit_backbone(args.size)
    sam3_vit = load_sam3_checkpoint_to_504(sam3_vit, sam3_path)
    
    sam3 = load_sam3(sam3_path, device)
    

    sam3.backbone.vision_backbone.trunk = sam3_vit

    model = LoRA_SAM3(sam3)
    # model = SAM3_DNet(sam_encoder)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model.eval()
    return model.to(device)


def run_inference(model, test_loader, device):
    """
    Run a forward pass for SAM3DUNet_V4.
    Handles RGB + edge input and upsamples output to match GT.
    """
    x, gt, name = test_loader.load_data()  # RGB + edge
    _, pred = model(x.to(device))

    # Upsample to match target size
    pred = F.interpolate(pred, size=gt.shape, mode='bilinear', align_corners=False)
    pred = torch.sigmoid(pred).cpu().squeeze().numpy()
    return pred, gt, name


def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    test_loader = TestDataset.TestDataset(args, args.size)

    # Model
    model = load_model(args.sam3_path, args.checkpoint, device)

    #########################################################
    # Evaluate model FLOPs and timing
    if evaluate:
        print_trainable_params(model)
        model_for_flops = copy.deepcopy(model)

        macs, params = get_model_complexity_info(
            model_for_flops, 
            (3, args.size, args.size),  # 4 channels: RGB + edge
            as_strings=True, 
            verbose=False
        )
        print(f"Computational complexity: {macs}, Parameters: {params}")
        del model_for_flops
        torch.cuda.empty_cache()

        dummy_input = torch.randn(1, 3, args.size, args.size).to(device)
        with torch.no_grad():
            for _ in range(20):  # warmup
                _ = model(dummy_input)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                _ = model(dummy_input)
            torch.cuda.synchronize()
            end = time.time()

        ms_per_image = (end - start) / 100 * 1000
        print(f"Runtime: {ms_per_image:.2f} ms per image")
    #########################################################

    # Metrics
    SM = Smeasure()
    EM = Emeasure()
    MAE_ = MAE()
    WF = WeightedFmeasure()
    FMv2 = py_sod_metrics.FmeasureV2(
        metric_handlers={
            "fm": py_sod_metrics.FmeasureHandler(with_adaptive=True, with_dynamic=True, beta=0.3),
            "iou": py_sod_metrics.IOUHandler(with_adaptive=True, with_dynamic=True),
        }
    )

    with torch.no_grad():
        with torch.autocast("cuda"):
            for i in tqdm(range(test_loader.length), desc="Evaluating"):
                pred, gt, name = run_inference(model, test_loader, device)

                # FMv2 expects float64
                FMv2.step(pred=pred.astype(np.float64), gt=gt.astype(np.float64))

                if binarize:
                    pred = (pred > THRESHOLD).astype(np.uint8) * 255

                pred_uint8 = (pred * 255).astype(np.uint8)

                SM.step(pred=pred_uint8, gt=gt)
                EM.step(pred=pred_uint8, gt=gt)
                MAE_.step(pred=pred_uint8, gt=gt)
                WF.step(pred=pred_uint8, gt=gt)

                if args.save_images:
                    fname = os.path.basename(name)
                    save_file = os.path.join(args.save_path, fname)
                    imageio.imsave(save_file, pred_uint8)

        # Aggregate metrics
        results = {
            "mIoU": FMv2.get_results()["iou"]["dynamic"].mean(),
            "Smeasure": SM.get_results()["sm"],
            "maxFm": FMv2.get_results()["fm"]["dynamic"].max(),
            "w_Fm": WF.get_results()["wfm"],
            "meanEm": EM.get_results()["em"]["curve"].mean(),
            "MAE": MAE_.get_results()["mae"]
        }

        print("\nEvaluation Results:")
        for k, v in results.items():
            print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    args.save_images = False
    args.size = 1008
    args.sam3_path = "./sam3-main/sam3.pt"
    args.checkpoint = "./checkpoints/LoRA-SAM3-best.pth"


    ### SOD ###
    # args.checkpoint = "./Saved_checkpoints/672/10_epochs/DUTS_HRSOD_UHRSD/LoRA-SAM3-best.pth"
    # args.task = "SOD"

    # args.path = "/mnt/c/Projects/datasets/RGB/DUTS-TE/"
    # args.save_path = "./test/SOD/DUTS-TE/"

    # args.path = "/mnt/c/Projects/datasets/RGB/DUT-OMRON/"
    # args.save_path = "./test/SOD/DUT-OMRON/"

    # args.path = "/mnt/c/Projects/datasets/RGB/ECSSD/"
    # args.save_path = "./test/SOD/ECSSD/"

    # args.path = "/mnt/c/Projects/datasets/RGB/HKU-IS/"
    # args.save_path = "./test/SOD/HKU-IS/"

    # args.path = "/mnt/c/Projects/datasets/RGB/PASCAL-S/"
    # args.save_path = "./test/SOD/PASCAL-S/"

    ### HR-SOD ###
    # args.checkpoint = "./Saved_checkpoints/DUTS_HRSOD_UHRSD/LoRA-SAM3-best.pth"
    # args.task = "SOD"

    # args.path = "/mnt/c/Projects/datasets/RGB/DAVIS-S/"
    # args.save_path = "./test/HR-SOD/DAVIS-S/"

    # args.path = "/mnt/c/Projects/datasets/RGB/HRSOD-TE/"
    # args.save_path = "./test/HR-SOD/HRSOD-TE/"

    # args.path = "/mnt/c/Projects/datasets/RGB/UHRSD-TE/"
    # args.save_path = "./test/HR-SOD/UHRSD-TE/"


    ### RGB-D SOD ###
    # args.checkpoint = "./Saved_checkpoints/NJU2K_NLPR/LoRA-SAM3-best.pth"
    # args.task = "SOD"

    # args.path = "/mnt/c/Projects/datasets/RGB-D_SOD/NJU2K/NJU2K_Test/"
    # args.save_path = "./test/RGB-D_SOD/NJU2K/"
    
    # args.path = "/mnt/c/Projects/datasets/RGB-D_SOD/NLPR/NLPR_Test/"
    # args.save_path = "./test/RGB-D_SOD/NLPR/"

    # args.path = "/mnt/c/Projects/datasets/RGB-D_SOD/SIP/"
    # args.save_path = "./test/RGB-D_SOD/SIP/"

    # args.path = "/mnt/c/Projects/datasets/RGB-D_SOD/STERE/"
    # args.save_path = "./test/RGB-D_SOD/STERE/"

    ### COD ###
    # args.checkpoint = "./Saved_checkpoints_ICIP_2026/672/20_epochs/TrainDataset/LoRA-SAM3-best.pth"
    # args.checkpoint = "./Ablations/No_AttGate/LoRA-SAM3-best.pth"

    # args.path = "/mnt/c/Projects/datasets/COD/TrainDataset/"
    # args.save_path = "./test/COD/TrainDataset/"

    # args.path = "/mnt/c/Projects/datasets/COD/CAMO/Test/"
    # args.save_path = "./test/COD/CAMO/"

    # args.path = "/mnt/c/Projects/datasets/COD/COD10K/Test/"
    # args.save_path = "./test/COD/COD10K/"

    # args.path = "/mnt/c/Projects/datasets/COD/CHAMELEON/Test/"
    # args.save_path = "./test/COD/CHAMELEON/"

    # args.path = "/mnt/c/Projects/datasets/COD/NC4K/"
    # args.save_path = "./test/COD/NC4K/"

    ### MAS ###
    # args.checkpoint = "./Saved_checkpoints/672/20_epochs/MAS3K_RMAS_COD/LoRA-SAM3-best.pth"
    # args.checkpoint = "./Ablations/AttGate+LoRA=8/LoRA-SAM3-best.pth"


    args.path = "/mnt/c/Projects/datasets/MAS/MAS3K/Test/"
    args.save_path = "./test/MAS/MAS3K/"

    # args.path = "/mnt/c/Projects/datasets/MAS/RMAS/Test/"
    # args.save_path = "./test/MAS/RMAS/"

    




    main(args)
