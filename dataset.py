import os
import random
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from glob import glob

class AddNoise(object):
    """Add SNR noise [-1, 1]
    """

    def __init__(self, noise, snr_levels=[-5, 0, 5, 10, 15, 20, 9999]):
        assert noise.dtype in [torch.float32, torch.float64], f"noise only supports float data type, not {noise.dtype}"
        
        self.noise = noise
        self.snr_levels = snr_levels

    def get_power(self, clip):
        clip = clip **2
        return torch.sum(clip) / (len(clip) * 1.0)

    def __call__(self, signal):
        assert signal.dtype in [torch.float32, torch.float64], f"signal only supports float32 data type, not {signal.dtype}"
        snr_target = random.choice(self.snr_levels)
        if snr_target == 9999:
            return signal
        else:
            # -- get noise
            start_idx = random.randint(0, len(self.noise)-len(signal))
            noise_clip = self.noise[start_idx:start_idx+len(signal)]

            sig_power = self.get_power(signal)
            noise_clip_power = self.get_power(noise_clip)
            factor = (sig_power / noise_clip_power ) / (10**(snr_target / 10.0))
            desired_signal = (signal + noise_clip*torch.sqrt(factor)).float()
            return desired_signal

class NDCME(Dataset):
    def __init__(self, video_dirpath, audio_dirpath, mode) -> None:
        super().__init__()
        self.dirpath = {'video': video_dirpath, 'audio': audio_dirpath}
        self.mode = mode

        self.vids = glob(os.path.join(self.dirpath['video'], '*'))
        self.wavs = glob(os.path.join(self.dirpath['audio'], '*'))
        self.expr2int = {'Laughs':0, 'Smiles':1, 'None':2}

        np.random.seed(0)
        np.random.shuffle(self.vids)
        self.fix_modality_mismatch()

        preprocessing = [
            T.Resize((96, 96), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop((88, 88)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.5452, 0.3590, 0.2160],
                std=[0.2193, 0.1876, 0.1488]
            )
        ]

        if mode == 'train':
            # take 70% of the data for training
            self.vids = self.vids[:int(0.7*len(self.vids))]
            self.wavs = self.wavs[:int(0.7*len(self.wavs))]
            preprocessing.append(T.RandomHorizontalFlip())
            self.noise = AddNoise(
                torch.Tensor(np.load(
                    os.path.join(os.path.dirname(self.dirpath['video']), 'babbleNoise_resample_16K.npy')
                ))
            )
        elif mode == 'val':
            # take 15% of the data for validation
            self.vids = self.vids[int(0.7*len(self.vids)):int(0.85*len(self.vids))]
            self.wavs = self.wavs[int(0.7*len(self.wavs)):int(0.85*len(self.wavs))]
        elif mode == 'test':
            # take 15% of the data for testing
            self.vids = self.vids[int(0.85*len(self.vids)):]
            self.wavs = self.wavs[int(0.85*len(self.wavs)):]
        
        expr_count = [
            np.count_nonzero(['Laughs' in vid for vid in self.vids]),
            np.count_nonzero(['Smiles' in vid for vid in self.vids]),
            np.count_nonzero(['None' in vid for vid in self.vids])
        ]

        self.weighted_sampling = [
            1/expr_count[0] if 'Laughs' in vid 
            else 1/expr_count[1] if 'Smiles' in vid
            else 1/expr_count[2] for vid in self.vids
        ]

        self.preprocess = T.Compose(preprocessing)
        self.norm_mean=-7.2984
        self.norm_std=3.2540

    def fix_modality_mismatch(self):
        files_to_keep = []
        vids_filename = [os.path.basename(vid) for vid in self.vids]
        wavs_filename = [os.path.basename(wav) for wav in self.wavs]
        for vid in vids_filename:
            if vid in wavs_filename:
                files_to_keep.append(vid)
        
        self.vids = [os.path.join(self.dirpath['video'], filename) for filename in files_to_keep]
        self.wavs = [os.path.join(self.dirpath['audio'], filename) for filename in files_to_keep]
        assert len(self.vids) == len(self.wavs)

    def __len__(self):
        return len(self.vids)
    
    def __getitem__(self, index):
        vid = self.vids[index]
        frames = np.load(vid, allow_pickle=True)['data']
        frames = [self.preprocess(Image.fromarray(frame)) for frame in frames]
        frames = torch.stack(frames)
        frames = torch.einsum('tchw->chwt', frames)
        target_length = 29
        n_frames = frames.shape[-1]
        p = target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ConstantPad1d((0, p), 0)
            frames = m(frames)
        elif p < 0:
            frames = frames[:, :, :, 0:frames]

        wav = self.wavs[index]
        waveform = np.load(wav, allow_pickle=True)['data']
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform - waveform.mean()

        ## Since MSTCN uses raw audio
        fbank = waveform
        target_length = 19456
        n_frames = fbank.shape[-1]
        p = target_length - n_frames
        
        if p > 0:
            m = torch.nn.ConstantPad1d((0, p), 0)
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[:, 0:target_length]
            
        if self.mode == 'train':
            fbank = self.noise(fbank)

        label, intensity = self.get_label_from_filename(vid)
        label = self.expr2int[label]
        return frames, fbank, label
    
    def get_label_from_filename(self, filename):
        return filename.split('_')[4:6]