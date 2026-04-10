import cv2, os, glob, random, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset
import torch

# ----------------------
# STEP 1: Load individual tiles
# ----------------------
def load_triplet(vv_path, vh_path, label_path, f):
    if not f.lower().endswith(".png"):
        return None
    name = os.path.splitext(f)[0]
    vv_file = os.path.join(vv_path, f"{name}_vv.png")
    vh_file = os.path.join(vh_path, f"{name}_vh.png")
    lbl_file = os.path.join(label_path, f)
    if not (os.path.exists(vv_file) and os.path.exists(vh_file) and os.path.exists(lbl_file)):
        return None

    vv = cv2.imread(vv_file, cv2.IMREAD_GRAYSCALE)
    vh = cv2.imread(vh_file, cv2.IMREAD_GRAYSCALE)
    lbl = cv2.imread(lbl_file, cv2.IMREAD_GRAYSCALE)
    if vv is None or vh is None or lbl is None:
        return None

    vv = vv.astype(np.float32)
    vh = vh.astype(np.float32)
    lbl = (lbl > 0).astype(np.uint8)
    return vv, vh, lbl

# ----------------------
# STEP 2: Load full dataset from a path
# ----------------------
def load_dataset(path, max_workers=8):
    samples = []
    dir_list = sorted(os.listdir(path))
    print(f"Found directories: {dir_list}")

    for dir_name in dir_list:
        vv_path = os.path.join(path, dir_name, "tiles/vv")
        vh_path = os.path.join(path, dir_name, "tiles/vh")
        label_path = os.path.join(path, dir_name, "tiles/flood_label")
        if not os.path.exists(label_path):
            continue

        file_list = sorted(os.listdir(label_path))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(lambda f: load_triplet(vv_path, vh_path, label_path, f), file_list)
            )
        results = [r for r in results if r is not None]

        for vv, vh, lbl in results:
            samples.append({
                "vv": vv,
                "vh": vh,
                "label": lbl,
                "event": dir_name
            })
    print(f"Total samples loaded from {path}: {len(samples)}")
    return samples

# ----------------------
# STEP 3: Split dataset by event (train/val)
# ----------------------
def split_by_event(samples, split_ratio=0.8, seed=42):
    events = sorted(list(set(s["event"] for s in samples)))
    rng = np.random.default_rng(seed)
    rng.shuffle(events)
    split = int(len(events) * split_ratio)
    train_events = set(events[:split])
    val_events = set(events[split:])
    train_samples = [s for s in samples if s["event"] in train_events]
    val_samples   = [s for s in samples if s["event"] in val_events]
    print(f"Train events: {len(train_events)}, Validation events: {len(val_events)}")
    print(f"Train tiles: {len(train_samples)}, Validation tiles: {len(val_samples)}")
    return train_samples, val_samples

# ----------------------
# STEP 4: Compute global training statistics
# ----------------------
def compute_train_stats(samples, use_median_on="vv", eps=1e-6):
    VV = np.stack([s["vv"] for s in samples])
    VH = np.stack([s["vh"] for s in samples])
    log_ratio = np.log((VV + eps) / (VH + eps))

    # Compute median-filtered version
    if use_median_on == "vv":
        med = np.stack([cv2.medianBlur(vv.astype(np.float32), 3) for vv in VV])
    else:
        med = np.stack([cv2.medianBlur(vh.astype(np.float32), 3) for vh in VH])

    stats = {
        "vv_mean": VV.mean(),
        "vv_std": VV.std(),
        "vh_mean": VH.mean(),
        "vh_std": VH.std(),
        "lr_mean": log_ratio.mean(),
        "lr_std": log_ratio.std(),
        "med_mean": med.mean(),
        "med_std": med.std()
    }
    print("Training stats computed including median.")
    return stats

# ----------------------
# STEP 5: Compute features using TRAIN stats
# ----------------------
def compute_sar_features(vv, vh, stats, eps=1e-6, use_median_on="vv"):
    log_ratio = np.log((vv + eps) / (vh + eps))

    vv_n = (vv - stats["vv_mean"]) / stats["vv_std"]
    vh_n = (vh - stats["vh_mean"]) / stats["vh_std"]
    lr_n = (log_ratio - stats["lr_mean"]) / stats["lr_std"]

    if use_median_on == "vv":
        med = cv2.medianBlur(vv.astype(np.float32), 3)
    else:
        med = cv2.medianBlur(vh.astype(np.float32), 3)
    med_n = (med - stats["med_mean"]) / stats["med_std"]

    return np.stack([vv_n, vh_n, lr_n, med_n], axis=0).astype(np.float32)


# ----------------------
# STEP 6: PyTorch dataset
# ----------------------
class FloodSARSegDataset(Dataset):
    def __init__(self, samples, stats, use_median_on="vv"):
        self.samples = samples
        self.stats = stats
        self.use_median_on = use_median_on

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        vv, vh, lbl = s["vv"], s["vh"], s["label"].astype(np.int64)
        sar_feat = compute_sar_features(vv, vh, self.stats, use_median_on=self.use_median_on)
        return {
            "sar": torch.tensor(sar_feat, dtype=torch.float32),
            "label": torch.tensor(lbl, dtype=torch.long)
        }

# ----------------------
# STEP 7: Final Usage
# ----------------------
def prepare_datasets(path_train, path_test, val_split=0.2):
    # Load training data
    train_samples_all = load_dataset(path_train)
    train_samples, val_samples = split_by_event(train_samples_all, split_ratio=1.0 - val_split)

    # Compute train stats
    stats = compute_train_stats(train_samples)

    # Load test set
    test_samples = load_dataset(path_test)

    # Wrap in datasets
    train_dataset = FloodSARSegDataset(train_samples, stats)
    val_dataset   = FloodSARSegDataset(val_samples, stats)
    test_dataset  = FloodSARSegDataset(test_samples, stats)

    return train_dataset, val_dataset, test_dataset, stats

#----------------------------------------------------------Augmentation-----------------------------------------------------------------------#
def rotate_image(img, angle, is_label=False):
    H, W = img.shape
    M = cv2.getRotationMatrix2D((W//2, H//2), angle, 1.0)
    flags = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (W, H), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def flip_image(img, mode):
    if mode == 'h': return np.fliplr(img)
    elif mode == 'v': return np.flipud(img)
    return img

def apply_augmentation(vv, vh, lbl, aug_type):
    """
    Apply a single augmentation type to VV, VH, and label.
    """
    if aug_type.startswith('rotate'):
        angle = int(aug_type.replace('rotate', ''))
        vv_aug = rotate_image(vv, angle, is_label=False)
        vh_aug = rotate_image(vh, angle, is_label=False)
        lbl_aug = rotate_image(lbl, angle, is_label=True)
    elif aug_type == 'flip_h':
        vv_aug = flip_image(vv, 'h')
        vh_aug = flip_image(vh, 'h')
        lbl_aug = flip_image(lbl, 'h')
    elif aug_type == 'flip_v':
        vv_aug = flip_image(vv, 'v')
        vh_aug = flip_image(vh, 'v')
        lbl_aug = flip_image(lbl, 'v')
    else:
        vv_aug, vh_aug, lbl_aug = vv.copy(), vh.copy(), lbl.copy()
    return vv_aug, vh_aug, lbl_aug

def prepare_augmentation_indices(bin_indices, non_flood_count, current_counts, needed_aug=None):
    """
    Compute which images to augment to balance classes.
    """
    total_current_flood = sum(current_counts.values())
    if needed_aug is None:
        needed_aug = non_flood_count - total_current_flood

    # Compute bin weights: bins with fewer samples get higher weight
    bin_weights = {b: 1.0 / max(1, cnt) for b, cnt in current_counts.items()}
    total_weight = sum(bin_weights.values())
    bin_probs = {b: w / total_weight for b, w in bin_weights.items()}

    # Select images to augment
    augmented_indices = []
    bin_list = list(bin_indices.keys())
    for _ in range(needed_aug):
        chosen_bin = random.choices(bin_list, weights=[bin_probs[b] for b in bin_list])[0]
        img_idx = random.choice(bin_indices[chosen_bin])
        augmented_indices.append(img_idx)

    return augmented_indices, bin_probs

def augment_dataset(VV_train, VH_train, Label_train, events, augmented_indices,
                    aug_types=['rotate90','rotate180','rotate270','flip_h','flip_v'], augmentation = 'aug'):
    samples_aug = []
    # original samples
    for i in range(len(VV_train)):
        samples_aug.append({"vv": VV_train[i],"vh": VH_train[i],"label": Label_train[i],"event": events[i],"is_aug": False})

    if str(augmentation).strip().lower() == 'aug':
        print('Applying augmentation ...')
        for idx in augmented_indices:
            aug_type = random.choice(aug_types)
            vv_aug, vh_aug, lbl_aug = apply_augmentation(VV_train[idx], VH_train[idx], Label_train[idx], aug_type)
            samples_aug.append({"vv": vv_aug, "vh": vh_aug, "label": lbl_aug, "event": events[idx], "is_aug": True})
    return samples_aug
