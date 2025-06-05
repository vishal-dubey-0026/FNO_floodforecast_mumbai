import os, glob, numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from torch.utils.data import Dataset, DataLoader
import torch

from affine import Affine

# flood_transform = Affine(30.0, 0.0, 337127.9191328756,
#                   0.0, -30.0, 3192398.1353707141)

flood_transform = None
flood_pixel_size = 30
seq_len = 20
not_valid_val = -3.4028e+38
MAX_TIME_STEPS = 5000
MAX_DEM = 5000
MAX_FLOOD_DEPTH = 100
MAX_RAINFALL = 100








# 2. Define paths
RAW_FLOOD_DIR     = "/content/FloodCastBench/FloodCastBench/High-fidelity flood forecasting/30m/Australia"
ALIGNED_FLOOD_DIR = "/content/FloodCastBench/FloodCastBench/Aligned_Flood_Australia"
DEM_TIF           = "/content/FloodCastBench/FloodCastBench/Relevant data/DEM/Australia_DEM.tif"
LULC_TIF          = "/content/FloodCastBench/FloodCastBench/Relevant data/Land use and land cover/Australia.tif"
RAIN_DIR          = "/content/FloodCastBench/FloodCastBench/Relevant data/Rainfall/Australia flood"
os.makedirs(ALIGNED_FLOOD_DIR, exist_ok=True)

# 3. Read DEM once as the master reference
with rasterio.open(DEM_TIF) as ref:
    ref_crs       = ref.crs
    ref_transform = ref.transform
    #############################
    # original transform = Affine(30, 0, c, 0, -30, f)
    orig = ref_transform
    c, f = orig.c, orig.f   # upper‐left corner coordinates
    # 1) compute the DEM’s full extent
    xmin, ymax = (c, f)
    # 3) build a 480 m affine
    flood_transform = Affine(
        flood_pixel_size, 0,   xmin,
        0,  -flood_pixel_size, ymax
    )
    #########################
    ref_width     = ref.width
    ref_height    = ref.height
    out_meta      = ref.meta.copy()
    out_meta.update({
        "crs":       ref_crs,
        "transform": flood_transform,
        "width":     ref_width,
        "height":    ref_height,
        "dtype":     np.float32,
    })

with rasterio.open(sorted(glob.glob(os.path.join(RAW_FLOOD_DIR,"*.tif")))[0]) as ref:
    ref_width     = ref.width
    ref_height    = ref.height
    out_meta.update({
        "width":     ref_width,
        "height":    ref_height,
    })



out_npy_raw = "/content/data/Australia_RAW.npy"


out_npy = "/content/data/Australia.npy"

# 6. Define the Dataset & DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



from typing import Optional
import random

TEMPORAL_RESOLUTION = 300 # in seconds
SPATIAL_DOWNSAMPLING_FACTOR = 1 
SPATITAL_RESOLUTION = flood_pixel_size * SPATIAL_DOWNSAMPLING_FACTOR # in meters
PATCH_H, PATCH_W = 256, 256


class FNOPlusLazyRainDataset(Dataset):
    def __init__(self, flood_npy, dem_tif, lulc_tif, rain_dir, seq_len=20, train_test_split = 0.7, feature_normalize = True):
        # reference grid from the DEM
        with rasterio.open(dem_tif) as ref:
            self.H_ref, self.W_ref    = out_meta["height"], out_meta["width"]
            self.ref_transform       = flood_transform
            self.ref_crs             = ref.crs

        # flood time‐series
        self.flood   = np.load(flood_npy).astype(np.float32)
        T, _, _      = self.flood.shape
        self.total_steps = MAX_TIME_STEPS
        self.seq_len = seq_len
        self.train_test_split = train_test_split
        self.feature_normalize = feature_normalize

        # DEM & LULC resampled to reference grid
        self.dem  = self.load_and_resample(dem_tif, Resampling.bilinear)[None]
        self.dem = self.process_dem(self.dem)


        raw_lulc   = self.load_and_resample(lulc_tif, Resampling.nearest).astype(int)
        self.lulc  = raw_lulc[None].astype(np.float32)

        # Manning’s n from LULC
        lut = {
                1: 0.0350,   # water
                2: 0.1200,   # trees
                4: 0.0800,   # flooded vegetation
                5: 0.0350,   # crops
                7: 0.3750,   # built / urban
                8: 0.0265,   # bare ground
                11: 0.0375   # rangeland
            }
        self.manning  = np.vectorize(lut.get)(raw_lulc)[None].astype(np.float32)

        self.manning[np.isnan(self.manning)] = -1   # or any fallback you prefer



        # Rainfall: load & resample once

        rain_files = sorted(glob.glob(os.path.join(rain_dir, "*.tif")))
        rains = [self.load_and_resample(rf, Resampling.bilinear) for rf in rain_files]
        for idx in range(len(rains)):
            rains[idx][rains[idx] < 0] = 0

        # map flood steps → rain index
        idx_map     = np.minimum(np.arange(T)//6, len(rains)-1)
        self.rains_map = [rains[i] for i in idx_map]
        self.spatial_coords = self.add_coord_channels(H = self.H_ref, W = self.W_ref)
        self.space_time_coords = self.add_X_Y_T_channels(H = self.H_ref, W = self.W_ref)


    def load_and_resample(self, path, resampling):
        # open source
        with rasterio.open(path) as src:
            src_data      = src.read(1)
            src_transform = src.transform
            src_crs       = src.crs if src.crs is not None else self.ref_crs

        dst = np.zeros((self.H_ref, self.W_ref), dtype=np.float32)

        dst = src_data

        # reproject(
        #     source        = src_data,
        #     destination   = dst,
        #     src_transform = src_transform,
        #     src_crs       = src_crs,
        #     dst_transform = self.ref_transform,
        #     dst_crs       = self.ref_crs,
        #     resampling    = resampling
        # )


        return dst.astype(np.float32)

    def add_coord_channels(self, H, W):

        # X-,Y- grids  (normalised 0-1)
        col = np.linspace(0, 1, W, endpoint=False)
        row = np.linspace(0, 1, H, endpoint=False)[:, None]
        Xc = np.tile(col, (H,1)).astype(np.float32)
        Yc = np.tile(row, (1,W)).astype(np.float32)

        return np.stack([Xc, Yc], axis=0)

    def add_time_channels(self, step, total_steps, H, W):
        # T grid (constant)
        T_norm = step / (total_steps - 1)
        Tc = np.full((H, W), T_norm, dtype=np.float32)[None]
        return Tc

    def add_X_Y_T_channels(self, H, W):
        t0 = self.seq_len
        lmbdleft, lmbdright = 0, (W - 1) * SPATITAL_RESOLUTION
        thtlower, thtupper = 0, (H - 1) * SPATITAL_RESOLUTION
        dt = TEMPORAL_RESOLUTION
        t00, tfinal = 0, (t0 - 1) * dt
        m = W
        n = H

        t = np.linspace(t00, tfinal, t0)
        x = np.linspace(lmbdleft, lmbdright, m)
        y = np.linspace(thtlower, thtupper, n)
        data_star = np.hstack((x.flatten(), y.flatten(), t.flatten()))
        # Data normalization
        lb = data_star.min(0)
        ub = data_star.max(0)

        gridx = torch.from_numpy(x)
        gridx = gridx.reshape(1, m, 1, 1, 1).repeat([1, 1, n, t0, 1])
        #gridx = gridx.reshape(1, 1, m, 1, 1).repeat([1, n, 1, t0, 1])
        gridy = torch.from_numpy(y)
        gridy = gridy.reshape(1, 1, n, 1, 1).repeat([1, m, 1, t0, 1])
        #gridy = gridy.reshape(1, n, 1, 1, 1).repeat([1, 1, m, t0, 1])
        gridt = torch.from_numpy(t)
        gridt = gridt.reshape(1, 1, 1, t0, 1).repeat([1, m, n, 1, 1]) # BHWTC
        input_data = torch.cat((gridt, gridx, gridy), dim=-1)
        input_data = 2.0 * (input_data - lb) / (ub - lb) - 1.0

        return torch.squeeze(input_data, dim = 0).float() # BHWTC -> HWTC

    def random_rotate_flip(self, x, y):
        """
        x: np.ndarray  (C, H, W)
        y: np.ndarray  (seq_len, H, W)
        Returns augmented (x, y) with identical ops applied to both.
        """
        # 1 – random 90° rot k∈{0,1,2,3}
        k = np.random.randint(0, 4)
        if k:
            x = np.rot90(x, k, axes=(1, 2))      # rotate over H-W axes
            y = np.rot90(y, k, axes=(1, 2))

        # 2 – random horizontal flip
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=2)               # flip W dimension
            y = np.flip(y, axis=2)

        # 3 – random vertical flip
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=1)               # flip H dimension
            y = np.flip(y, axis=1)

        return x.copy(), y.copy()                # .copy() → contiguous arrays


    def augment_x_y(self, x, y, seq_len, manning_idx, depth_idx, dem_idx, spatial_coord_idx, jitter_xy=True, scale_rain=True, manning_scale = True, add_depth_noise = False, add_dem_noise = True):
        # --- rainfall intensity scale ---
        if scale_rain and np.random.rand() < 0.5:
            factor = np.random.normal(loc = 0, scale = 0.001) # 0.1 mm

            x[-seq_len:] = (x[-seq_len:] + factor).clip(0, None)

        if add_depth_noise and np.random.rand() < 0.3:
            # depth noise
            noise = np.random.normal(0, 0.02, x[depth_idx].shape) # 0.02 m
            x[depth_idx] = (x[depth_idx] + noise).clip(0, None)


        # --- sub-pixel XY jitter ---
        if jitter_xy and np.random.rand() < 0.5:
            dx, dy = np.random.uniform(-0.5, 0.5, 2) # 0.5 in pixel units
            x_c_idx, y_c_idx = spatial_coord_idx
            x[x_c_idx] = (x[x_c_idx] + (dx * 1/self.W_ref)).clip(0, None)      # X-coord channel, [0 - 1] scale jitter
            x[y_c_idx] = (x[y_c_idx] + (dy * 1/self.H_ref)).clip(0, None)      # Y-coord channel, [0 - 1] scale jitter

        if manning_scale and np.random.rand() < 0.5:
            x[manning_idx] *= np.random.uniform(0.9, 1.1) # 0.9x to 1.1x

        if add_dem_noise and np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.1, x[dem_idx].shape) # 0.1 m
            x[dem_idx] = (x[dem_idx] + noise)   #.clip(None, None) #dem can be negative

        return x.astype(np.float32), y.astype(np.float32)


    def fix_missing_values(self, tile: np.ma.MaskedArray,
                          masked_value_offset: Optional[float] = 30):
        tile.data[tile.mask] = masked_value_offset + np.max(tile)
        return tile

    def process_dem(self, dem_map):
        # with open(dem_tif_path, 'rb') as f:
        #     tiffbytes = f.read()
        np_ma_map = np.ma.masked_array(dem_map, mask=(dem_map < -2000))
        np_ma_map = self.fix_missing_values(np_ma_map)
        dem = np_ma_map.data
        return dem


    def numpy_to_torch_chw_to_hwtc(self, np_array):
        return torch.from_numpy(np_array).unsqueeze(dim = 0).repeat([self.seq_len, 1, 1, 1]).permute(2, 3, 0, 1) #CHW -> TCHW -> HWTC


    
    def random_spatial_crop(self, input_tensor, output_tensor, crop_h, crop_w):
        """
        Given:
          input_tensor:  torch.Tensor of shape (H, W, T, C_in)
          output_tensor: torch.Tensor of shape (H, W, T_out, C_out)
          crop_h, crop_w: spatial crop size

        Returns a single random spatial patch of size (crop_h, crop_w, T, C)
        from both input and output, cropping along H & W but keeping T and C intact.
        """
        H, W, _, _ = input_tensor.shape
        assert output_tensor.shape[0] == H and output_tensor.shape[1] == W, \
            "input and output must share the same H, W"

        # pick top-left corner uniformly
        max_i = H - crop_h
        max_j = W - crop_w
        if max_i < 0 or max_j < 0:
            raise ValueError(f"crop size ({crop_h},{crop_w}) is larger than tensor size ({H},{W})")

        i = random.randint(0, max_i)
        j = random.randint(0, max_j)

        input_patch  = input_tensor[ i : i + crop_h,
                                    j : j + crop_w,
                                    : , : ]   # yields (crop_h, crop_w, T, C_in)
        output_patch = output_tensor[i : i + crop_h,
                                    j : j + crop_w,
                                    : , : ]   # yields (crop_h, crop_w, T_out, C_out)

        return input_patch, output_patch

    def grid_spatial_patches(self, input_tensor, output_tensor, crop_h, crop_w):
        """
        Splits the full (H, W) domain into a grid of non-overlapping (crop_h, crop_w) patches.
        Only full patches are returned (any remainder along H or W is discarded).

        Returns:
          inp_patches  : Tensor of shape (N, crop_h, crop_w, T, C_in)
          oup_patches  : Tensor of shape (N, crop_h, crop_w, T_out, C_out)
          where N = (H // crop_h) * (W // crop_w)
        """
        H, W, T, C_in   = input_tensor.shape
        Ho, Wo, To, C_out = output_tensor.shape
        assert (Ho, Wo) == (H, W), "input and output must share same H, W"

        n_h = H // crop_h    # number of full patches along height
        n_w = W // crop_w    # number of full patches along width
        N   = n_h * n_w

        inp_patches  = []
        oup_patches  = []

        for i in range(n_h):
            for j in range(n_w):
                top = i * crop_h
                left = j * crop_w
                inp_patch = input_tensor[ top : top + crop_h,
                                          left: left + crop_w,
                                          :, : ]   # (crop_h, crop_w, T, C_in)
                oup_patch = output_tensor[top : top + crop_h,
                                          left: left + crop_w,
                                          :, : ]   # (crop_h, crop_w, T_out, C_out)

                inp_patches.append(inp_patch.unsqueeze(0))   # (1, crop_h, crop_w, T, C_in)
                oup_patches.append(oup_patch.unsqueeze(0))   # (1, crop_h, crop_w, T_out, C_out)

        inp_patches = torch.cat(inp_patches, dim=0)   # (N, crop_h, crop_w, T, C_in)
        oup_patches = torch.cat(oup_patches, dim=0)   # (N, crop_h, crop_w, T_out, C_out)

        return inp_patches, oup_patches


    def __len__(self):
        return (int(self.train_test_split * len(self.flood)) - self.seq_len) #// self.seq_len

  

    def __getitem__(self, idx):
        #idx = idx * self.seq_len
        depth    = self.flood[idx:idx+1] #CHW
        depth = self.numpy_to_torch_chw_to_hwtc(depth)
        dem = self.numpy_to_torch_chw_to_hwtc(self.dem)
        manning = self.numpy_to_torch_chw_to_hwtc(self.manning)

        
        rain_seq = np.stack(self.rains_map[idx:idx+self.seq_len], axis=0)
        rain_seq = torch.from_numpy(rain_seq).unsqueeze(dim = 1).permute(2, 3, 0, 1) #THW -> TCHW -> HWTC
        x = torch.cat([depth, self.space_time_coords, dem, manning, rain_seq], dim = -1)
        #x        = np.concatenate([depth, self.spatial_coords, time_domain_channel, self.dem, self.manning, rain_seq], axis=0)
        y        = self.flood[idx:idx+self.seq_len].astype(np.float32)
        y = torch.from_numpy(y).unsqueeze(dim = 1).permute(2, 3, 0, 1) #THW -> TCHW -> HWTC
        x, y = self.random_spatial_crop(input_tensor = x, output_tensor = y, crop_h = PATCH_H, crop_w = PATCH_W)
        
        #x, y = self.augment_x_y(x = x, y = y, seq_len = self.seq_len, manning_idx = 5, depth_idx = 0, dem_idx = 4, spatial_coord_idx = (1, 2))
       

        return x, y


class FNOPlusTestDataset(FNOPlusLazyRainDataset):
    """Produces non-overlapping seq_len blocks for evaluation."""
    def __len__(self):
        return (int((1 - self.train_test_split) * len(self.flood)) - self.seq_len) #// self.seq_len

    def __getitem__(self, idx):
        offset = int(self.train_test_split * len(self.flood)) - self.seq_len
        idx = offset + idx #* self.seq_len + 
        depth    = self.flood[idx:idx+1] #CHW
        depth = self.numpy_to_torch_chw_to_hwtc(depth)
        dem = self.numpy_to_torch_chw_to_hwtc(self.dem)
        manning = self.numpy_to_torch_chw_to_hwtc(self.manning)

        
        rain_seq = np.stack(self.rains_map[idx:idx+self.seq_len], axis=0)
        rain_seq = torch.from_numpy(rain_seq).unsqueeze(dim = 1).permute(2, 3, 0, 1) #THW -> TCHW -> HWTC
        x = torch.cat([depth, self.space_time_coords, dem, manning, rain_seq], dim = -1)
        y        = self.flood[idx:idx+self.seq_len].astype(np.float32)
        y = torch.from_numpy(y).unsqueeze(dim = 1).permute(2, 3, 0, 1) #THW -> TCHW -> HWTC      
        x, y = self.random_spatial_crop(input_tensor = x, output_tensor = y, crop_h = PATCH_H, crop_w = PATCH_W) 
        return x, y

# instantiate
train_ds = FNOPlusLazyRainDataset(
    flood_npy=out_npy_raw,
    dem_tif=DEM_TIF,
    lulc_tif=LULC_TIF,
    rain_dir=RAIN_DIR,
    seq_len=seq_len,
    train_test_split = 0.7,
    feature_normalize = True
)

# instantiate & test
test_ds = FNOPlusTestDataset(
    flood_npy=out_npy_raw,
    dem_tif=DEM_TIF,
    lulc_tif=LULC_TIF,
    rain_dir=RAIN_DIR,
    seq_len=seq_len,
    train_test_split = 0.7,
    feature_normalize = True
)


batch_size = 4
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)

test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

x_b, y_b = next(iter(train_loader))
x_b, y_b = x_b.to(device), y_b.to(device)
print("X shape:", x_b.shape, "on", x_b.device)
print("Y shape:", y_b.shape, "on", y_b.device)

x_b, y_b = next(iter(test_loader))
x_b, y_b = x_b.to(device), y_b.to(device)
print("X shape:", x_b.shape, "on", x_b.device)
print("Y shape:", y_b.shape, "on", y_b.device)




import torch.nn as nn
import torch


import torch.nn.functional as F


def add_padding(x, num_pad):
    if max(num_pad) > 0:
        res = F.pad(x, (num_pad[0], num_pad[1]), 'constant', 0)
    else:
        res = x
    return res


def add_padding2(x, num_pad1, num_pad2):
    if max(num_pad1) > 0 or max(num_pad2) > 0:
        res = F.pad(x, (num_pad2[0], num_pad2[1], num_pad1[0], num_pad1[1]), 'constant', 0.)
    else:
        res = x
    return res


def remove_padding(x, num_pad):
    if max(num_pad) > 0:
        res = x[..., num_pad[0]:-num_pad[1]]
    else:
        res = x
    return res


def remove_padding2(x, num_pad1, num_pad2):
    if max(num_pad1) > 0 or max(num_pad2) > 0:
        res = x[..., num_pad1[0]:-num_pad1[1], num_pad2[0]:-num_pad2[1]]
    else:
        res = x
    return res


def _get_act(act):
    if act == 'tanh':
        func = F.tanh
    elif act == 'gelu':
        func = F.gelu
    elif act == 'relu':
        func = F.relu_
    elif act == 'elu':
        func = F.elu_
    elif act == 'leaky_relu':
        func = F.leaky_relu_
    else:
        raise ValueError(f'{act} is not supported')
    return func


@torch.jit.script
def compl_mul3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixyz,ioxyz->boxyz", a, b)
    return res

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,3,4])
        
        z_dim = min(x_ft.shape[4], self.modes3)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[2], x_ft.shape[3], self.modes3, device=x.device, dtype=torch.cfloat)
        
        # if x_ft.shape[4] > self.modes3, truncate; if x_ft.shape[4] < self.modes3, add zero padding 
        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, :self.modes1, :self.modes2, :z_dim]
        out_ft[:, :, :self.modes1, :self.modes2, :] = compl_mul3d(coeff, self.weights1)
        
        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, -self.modes1:, :self.modes2, :z_dim]
        out_ft[:, :, -self.modes1:, :self.modes2, :] = compl_mul3d(coeff, self.weights2)
        
        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, :self.modes1, -self.modes2:, :z_dim]
        out_ft[:, :, :self.modes1, -self.modes2:, :] = compl_mul3d(coeff, self.weights3)
        
        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, -self.modes1:, -self.modes2:, :z_dim]
        out_ft[:, :, -self.modes1:, -self.modes2:, :] = compl_mul3d(coeff, self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(2), x.size(3), x.size(4)), dim=[2,3,4])
        return x



class FNN3d(nn.Module):
    def __init__(self, 
                 modes1, modes2, modes3,
                 width=16, 
                 fc_dim=128,
                 layers=None,
                 in_dim=5, out_dim=3,
                 act='gelu', 
                 pad_ratio=[0., 0.]):
        '''
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            fc_dim: dimension of fully connected layers
            in_dim: int, input dimension
            out_dim: int, output dimension
            act: {tanh, gelu, relu, leaky_relu}, activation function
            pad_ratio: the ratio of the extended domain
        '''
        super(FNN3d, self).__init__()

        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, 'Cannot add padding in more than 2 directions.'

        self.pad_ratio = pad_ratio
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.pad_ratio = pad_ratio

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv3d(
            in_size, out_size, mode1_num, mode2_num, mode3_num)
            for in_size, out_size, mode1_num, mode2_num, mode3_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.act = _get_act(act)

    def forward(self, x):
        '''
        Args:
            x: (batchsize, x_grid, y_grid, t_grid, 3)

        Returns:
            u: (batchsize, x_grid, y_grid, t_grid, 1)

        '''
        size_z = x.shape[-2]
        if max(self.pad_ratio) > 0:
            num_pad = [round(size_z * i) for i in self.pad_ratio]
        else:
            num_pad = [0., 0.]
        length = len(self.ws)
        batchsize = x.shape[0]
        
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = add_padding(x, num_pad=num_pad)
        size_x, size_y, size_z = x.shape[-3], x.shape[-2], x.shape[-1]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y, size_z)
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)
        x = remove_padding(x, num_pad=num_pad)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

"""
parser.add_argument('--layers', nargs='+', type=int, default=[16, 24, 24, 32, 32], help='Dimensions/layers of the NN')
parser.add_argument('--modes1', nargs='+', type=int, default=[8, 8, 12, 12], help='')
parser.add_argument('--modes2', nargs='+', type=int, default=[8, 8, 12, 12], help='')
parser.add_argument('--modes3', nargs='+', type=int, default=[8, 8, 8, 8], help='')
parser.add_argument('--fc_dim', type=int, default=128, help='')
parser.add_argument('--epochs', type=int, default=15000)
parser.add_argument('--activation', default='gelu', help='Activation to use in the network.')
"""
layers = [seq_len, 24, 24, 32, 32]
modes1 = [8, 8, 12, 12]
modes2 = [8, 8, 12, 12]
modes3 = [8, 8, 8, 8]
fc_dim = 128
activation = 'gelu'
C_in = 1 + (2 + 1) + 1 + 1 + 1   # depth + (spatial + time) + DEM + Manning + rain step
C_out = 1
#model = PINO2d(modes1=modes1, modes2=modes2, width=fc_dim, layers=layers, in_dim=C_in, out_dim=C_out).to(device)
model = FNN3d(modes1=modes1, modes2=modes2, modes3=modes3, fc_dim=fc_dim, layers=layers, in_dim=C_in, out_dim=C_out).to(device)




import numpy as np
import torch
import torch.nn.functional as F


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1) + 1e-5, self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1) + 1e-5, self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / (y_norms + 1e-5))
            else:
                return torch.sum(diff_norms / (y_norms + 1e-5))

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class LpLoss2(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1) + 1e-5, self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1) + 1e-5, self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / (y_norms + 1e-5))
            else:
                return torch.sum(diff_norms / (y_norms + 1e-5))

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.abs(x, y)

def GeoPC_loss(outputH, data_condition, init_condition):

    #data_loss
    h_gt = data_condition[0]

    # loss_d = loss_h + loss_qx + loss_qy
    loss = LpLoss(size_average=True)
    h_c = outputH
    # h_g = torch.unsqueeze(h_gt, dim=0)
    loss_h = loss(h_c, h_gt)
    loss_d = loss_h

    # if i == 0:
    #     _EPSILON = 1e-6
    h_init = init_condition[0]
    h_cc = outputH[:, 0, :, :]
    #     h_c = torch.squeeze(h_c)
    loss_c = loss(h_cc, h_init)

    return loss_d, loss_c



from re import M
import torch.nn.functional as F


def test():
    model.to(device).eval()

    # 2. Loss (we'll compute MSE manually below)
    criterion = nn.MSELoss(reduction="none")

    # 3. Initialize accumulators
    sum_rmse = 0.0
    sum_nse  = 0.0
    sum_r    = 0.0
    sum_csi_001 = 0.0
    sum_csi_01  = 0.0
    num_samples = 0

    # 4. Loop over your test_loader
    for x_batch, y_batch in test_loader:
        # Move data to GPU
        x_full = x_batch.to(device)
        y_full = y_batch.to(device)
        B, H, W, T, C = y_full.shape




        # Forward pass
        with torch.no_grad():
            pred = model(x_full) #BHWTC
        pred = F.threshold(pred, threshold=0, value=0)

        # Extract only the final forecast step (t=20)
        y_last = y_full
        p_last = pred

        B = p_last.size(0)
        # Flatten spatial dims
        t_flat = y_last.reshape(B, -1)
        p_flat = p_last.reshape(B, -1)

        # RMSE per sample
        mse_per  = ((p_last - y_last)**2).reshape(B, -1).mean(dim=1)
        rmse_per = torch.sqrt(mse_per)

        # NSE per sample
        sse = torch.sum((p_flat - t_flat)**2, dim=1)
        sst = torch.sum((t_flat - t_flat.mean(dim=1, keepdim=True))**2, dim=1)
        nse_per = 1 - sse / sst

        # Pearson r per sample
        t_cent = t_flat - t_flat.mean(dim=1, keepdim=True)
        p_cent = p_flat - p_flat.mean(dim=1, keepdim=True)
        r_per  = torch.sum(t_cent * p_cent, dim=1) / (
                    torch.sqrt(torch.sum(t_cent**2, dim=1)) *
                    torch.sqrt(torch.sum(p_cent**2, dim=1))
                )

        # CSI per sample at γ=0.001 and γ=0.01
        tp1 = ((p_flat>=0.001)&(t_flat>=0.001)).sum(dim=1).float()
        fp1 = ((p_flat>=0.001)&(t_flat< 0.001)).sum(dim=1).float()
        fn1 = ((p_flat< 0.001)&(t_flat>=0.001)).sum(dim=1).float()
        csi001 = tp1 / (tp1 + fp1 + fn1 + 1e-6)

        tp2 = ((p_flat>=0.01)&(t_flat>=0.01)).sum(dim=1).float()
        fp2 = ((p_flat>=0.01)&(t_flat< 0.01)).sum(dim=1).float()
        fn2 = ((p_flat< 0.01)&(t_flat>=0.01)).sum(dim=1).float()
        csi01  = tp2 / (tp2 + fp2 + fn2 + 1e-6)

        # Accumulate
        sum_rmse     += rmse_per.sum().item()
        sum_nse      += nse_per.sum().item()
        sum_r        += r_per.sum().item()
        sum_csi_001  += csi001.sum().item()
        sum_csi_01   += csi01.sum().item()
        num_samples  += B

    # 5. Compute and print averages
    avg_rmse    = sum_rmse    / num_samples
    avg_nse     = sum_nse     / num_samples
    avg_r       = sum_r       / num_samples
    avg_csi_001 = sum_csi_001 / num_samples
    avg_csi_01  = sum_csi_01  / num_samples

    print(f"Batchwise RMSE: {avg_rmse:.6f}")
    print(f"Batchwise NSE:  {avg_nse:.6f}")
    print(f"Batchwise Pearson r: {avg_r:.6f}")
    print(f"Batchwise CSI @0.001m: {avg_csi_001:.6f}")
    print(f"Batchwise CSI @0.01m:  {avg_csi_01:.6f}")


import torch
import torch.optim as optim
import torch.nn as nn

# 0. Pick your device (falls back to CPU if CUDA isn’t available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
model.load_state_dict(torch.load(f"/content/drive/MyDrive/FLOOD_NET_ML_MODEL/Pakistan-480m-sample/fnn3dplus_australia_v8_11.pth"))
model.train()

test_itr = 1

# 2. Define loss & optimizer
criterion = nn.MSELoss()
learning_rate = 5e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
total_train_itr = len(train_ds)/batch_size
# 3. Full training loop
num_epochs = 100
for epoch in range(12, num_epochs + 1):
    total_loss = 0.0
    sample_itr = 0
    for x_batch, y_batch in train_loader:
        sample_itr += 1
        # Move batch to GPU (non_blocking because pin_memory=True)
        x = x_batch.to(device, non_blocking=True)              
        y = y_batch.to(device, non_blocking=True)
        B, H, W, T, C = y.shape


        optimizer.zero_grad()
        #############
        h_init = x[..., 0, 0]
        init_condition = [h_init]
        data_condition = [y.permute(0, 4, 3, 1, 2)] # BHWTC -> BCTHW
    
        out = model(x)

        # print(out.shape)
        # boundary
        output = out.permute(0, 3, 1, 2, 4) # BHWTC -> BTHWC
        outputH = output[:, :, :, :, 0].clone()
        loss_d, loss_c = GeoPC_loss(outputH, data_condition, init_condition)
        loss = loss_c + loss_d
        ###########
      
        loss.backward()                  # gradients on GPU
        optimizer.step()                 # updates on GPU

        total_loss += loss.item()
        if sample_itr % 100 == 0:
            print(f"{sample_itr}/{total_train_itr}. loss is: {loss.item()}, loss_c: {loss_c.item()}, loss_d:{loss_d.item()}")

        del x, y, out, loss, loss_d, loss_c
        torch.cuda.empty_cache()
    avg = total_loss / len(train_loader)
    print(f"Epoch {epoch:02d}/{num_epochs} — Loss: {avg}")
    torch.save(model.state_dict(), f"/content/drive/MyDrive/FLOOD_NET_ML_MODEL/Pakistan-480m-sample/fnn3dplus_australia_v8_{epoch}.pth")
    if epoch % test_itr == 0:
        test()
        model.train()

# 4. Save the model (state dict lives on CPU by default, but that's fine)

print("Training complete and model saved.")
