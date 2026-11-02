import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import argparse
from tqdm import tqdm

# Project-specific imports
from mkunet_network import MK_UNet
from utils.dataloader_polyp import get_loader
from medpy.metric.binary import hd95

def dice_coefficient(predicted, labels):
    """Computes the Dice coefficient."""
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    total = predicted_flat.sum() + labels_flat.sum()
    return (2. * intersection + smooth) / (total + smooth)

def iou(predicted, labels):
    """Computes the Intersection over Union (IoU)."""
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    union = predicted_flat.sum() + labels_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def get_binary_metrics(pred, gt):
    """Calculates sensitivity, specificity, precision, and HD95."""
    tp = (pred * gt).sum().item()
    tn = ((1 - pred) * (1 - gt)).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()
    
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    
    try:
        if pred.sum() > 0 and gt.sum() > 0:
            hd_val = hd95(pred.cpu().numpy(), gt.cpu().numpy())
        else:
            hd_val = 100.0
    except:
        hd_val = 100.0
        
    return sensitivity, specificity, precision, hd_val

def test(model, path, dataset, opt, save_base=None):
    """Evaluates the model and saves prediction masks."""
    data_path = os.path.join(path, dataset)
    image_root = f'{data_path}/images/'
    gt_root = f'{data_path}/masks/'
    model.eval()
    
    test_loader = get_loader(
        image_root=image_root, gt_root=gt_root, 
        batchsize=opt.test_batchsize, trainsize=opt.img_size,
        shuffle=False, split='test', color_image=opt.color_image
    )
    
    DSC, IOU, total_images = 0.0, 0.0, 0
    detailed_results = []

    with torch.no_grad():
        for pack in tqdm(test_loader, desc=f"Inference on {dataset}"):
            images, gts, original_shapes, names = pack       
            images = images.cuda()
            gts = gts.cuda().float()

            ress = model(images)
            predictions = ress[0] if isinstance(ress, list) else ress
            
            for i in range(len(images)):
                h_orig, w_orig = int(original_shapes[0][i]), int(original_shapes[1][i])
                
                # 1. Prediction Resize (Bilinear)
                p = predictions[i].unsqueeze(0)
                pred_resized = F.interpolate(p, size=(h_orig, w_orig), mode='bilinear', align_corners=False).sigmoid().squeeze()
                
                # 2. Local Normalization
                pred_resized = (pred_resized - pred_resized.min()) / (pred_resized.max() - pred_resized.min() + 1e-8)
                
                # 3. GT Resize (Nearest)
                g = gts[i].unsqueeze(0)
                gt_resized = F.interpolate(g, size=(h_orig, w_orig), mode='nearest').squeeze()

                # 4. Binary Thresholding
                input_binary = (pred_resized >= 0.5).float()
                target_binary = (gt_resized >= 0.2).float()

                # 5. Metrics with original 4-decimal truncation logic
                d = dice_coefficient(input_binary, target_binary).item()
                io = iou(input_binary, target_binary).item()
                sens, spec, prec, hd = get_binary_metrics(input_binary, target_binary)

                DSC += d
                IOU += io
                total_images += 1

                # Store for Excel
                detailed_results.append({
                    'Name': names[i],
                    'Dice': d,
                    'IoU': io,
                    'Sensitivity': float('{:.4f}'.format(sens)),
                    'Specificity': float('{:.4f}'.format(spec)),
                    'Precision': float('{:.4f}'.format(prec)),
                    'HD95': float('{:.4f}'.format(hd))
                })

                # 6. Save Prediction Part
                if save_base:
                    pred_img = (input_binary.cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_base, names[i]), pred_img)

    return DSC / total_images, IOU / total_images, detailed_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True, help='ID of the run to test')
    parser.add_argument('--network', type=str, default='MK_UNet')
    parser.add_argument('--dataset_name', type=str, default='ClinicDB')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--test_batchsize', type=int, default=1)
    parser.add_argument('--color_image', default=True)
    parser.add_argument('--test_path', type=str, default='./data/polyp/target/')
    opt = parser.parse_args()

    # --- Paths ---
    save_base = f'./predictions_polyp/{opt.run_id}/{opt.dataset_name}/{opt.split}'
    os.makedirs(save_base, exist_ok=True)
    model_path = os.path.join(f'./model_pth/{opt.run_id}/', f'{opt.run_id}-best.pth')

    opt.test_path = f'{opt.test_path}/{opt.dataset_name}/'

    # --- Model Loading ---
    NET_CONFIGS = {
        'MK_UNet_T': [4, 8, 16, 24, 32],
        'MK_UNet_S': [8, 16, 32, 48, 80],
        'MK_UNet':   [16, 32, 64, 96, 160],
        'MK_UNet_M': [32, 64, 128, 192, 320],
        'MK_UNet_L': [64, 128, 256, 384, 512]
    }
    
    channels = NET_CONFIGS.get(opt.network, NET_CONFIGS['MK_UNet'])
    model = MK_UNet(num_classes=1, in_channels=3, channels=channels).cuda()
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    model.eval()

    # --- Run Inference ---
    mean_dice, mean_iou, results = test(model, opt.test_path, opt.split, opt, save_base=save_base)

    # --- Save to Excel ---
    df = pd.DataFrame(results)
    # Add average row
    mean_row = df.mean(numeric_only=True).to_dict()
    mean_row['Name'] = 'AVERAGE'
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    
    excel_name = f'results_polyp/Results_{opt.run_id}_{opt.dataset_name}_{opt.split}.xlsx'
    df.to_excel(excel_name, index=False)

    print(f"\nFinal Results for {opt.run_id}:")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Excel report saved to: {excel_name}")

    # --- Persistent Summary Logic ---
    summary_file = 'All_Runs_Summary_Polyp.xlsx'
    avg_data = {
            'run_id': opt.run_id,
            'network': opt.network,
            'dataset': opt.dataset_name,
            'split': opt.split,
            'dice': mean_dice,
            'iou': mean_iou,
            'sensitivity': mean_row['Sensitivity'],
            'specificity': mean_row['Specificity'],
            'precision': mean_row['Precision'],
            'HD95': mean_row['HD95']
    }
    df_summary_new = pd.DataFrame([avg_data])

    # Append to existing file or create a new one
    if os.path.exists(summary_file):
        # Read existing summary and append new row
        df_summary_existing = pd.read_excel(summary_file)
        df_summary_combined = pd.concat([df_summary_existing, df_summary_new], ignore_index=True)
        df_summary_combined.to_excel(summary_file, index=False)
    else:
        # Create new file
        df_summary_new.to_excel(summary_file, index=False) # Creates with header

    print(f"Summary appended to {summary_file}")