import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sogou_audio import fft, get_ana_win, get_syn_win


class SogouSTFT(torch.nn.Module):
    def __init__(self, fft_size=1024, hop_size=512):
        super(SogouSTFT, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size

        fourier_basis = fft(np.eye(self.fft_size), fft_size, hop_size).T
        fourier_basis = np.vstack([np.real(fourier_basis), np.imag(fourier_basis)])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(fourier_basis).T[:, None, :])

        ana_win = get_ana_win(fft_size, hop_size)
        syn_win = get_syn_win(fft_size, hop_size)

        forward_basis = torch.tensor(ana_win, dtype=torch.float32).view(1, 1, -1) * forward_basis
        inverse_basis = torch.tensor(syn_win, dtype=torch.float32).view(1, 1, -1) * inverse_basis

        self.register_buffer('forward_basis', forward_basis)
        self.register_buffer('inverse_basis', inverse_basis)

    def transform(self, x, keep_dc=True, transpose=False):
        '''
            transpose (Boolean):
                [batch_size, freq_bins, num_frames] -> [batch_size, num_frames, freq_bins]
            keep_dc (Boolean):
                remain DC
        '''
        batch_size = x.size(0)
        n_samples = x.size(1)

        x = x.view(batch_size, 1, n_samples)
        x = F.conv1d(x, self.forward_basis, stride=self.hop_size, padding=0)

        cut_off = self.fft_size // 2 + 1
        re = x[:, :cut_off, :]
        im = x[:, cut_off:, :]

        if not keep_dc:
            re = re[:, 1:, :]
            im = im[:, 1:, :]

        if transpose:
            re = re.transpose(1, 2)
            im = im.transpose(1, 2)
        return re, im

    def transform2(self, x, hop_size, keep_dc=True, transpose=False):
        '''
            transpose (Boolean):
                [batch_size, freq_bins, num_frames] -> [batch_size, num_frames, freq_bins]
            keep_dc (Boolean):
                remain DC
        '''
        batch_size = x.size(0)
        n_samples = x.size(1)

        x = x.view(batch_size, 1, n_samples)
        x = F.conv1d(x, self.forward_basis, stride=hop_size, padding=0)

        cut_off = self.fft_size // 2 + 1
        re = x[:, :cut_off, :]
        im = x[:, cut_off:, :]

        if not keep_dc:
            re = re[:, 1:, :]
            im = im[:, 1:, :]

        if transpose:
            re = re.transpose(1, 2)
            im = im.transpose(1, 2)
        return re, im

    def mag(self, x, keep_dc=True, transpose=False, eps=1e-5):
        '''
            transpose (Boolean):
                [batch_size, freq_bins, num_frames] -> [batch_size, num_frames, freq_bins]
            keep_dc (Boolean):
                remain DC
        '''
        re, im = self.transform(x, keep_dc, transpose)
        mag = (re ** 2 + im ** 2 + eps ** 2) ** 0.5
        return mag

    def mag2(self, x, hop_size, keep_dc=True, transpose=False, eps=1e-5):
        '''
            transpose (Boolean):
                [batch_size, freq_bins, num_frames] -> [batch_size, num_frames, freq_bins]
            keep_dc (Boolean):
                remain DC
        '''
        re, im = self.transform2(x, hop_size, keep_dc, transpose)
        mag = (re ** 2 + im ** 2 + eps ** 2) ** 0.5
        return mag

    def mag_angle(self, x, keep_dc=True, transpose=False, eps=1e-5):
        '''
            transpose (Boolean):
                [batch_size, freq_bins, num_frames] -> [batch_size, num_frames, freq_bins]
            keep_dc (Boolean):
                remain DC
        '''
        re, im = self.transform(x, keep_dc, transpose)
        mag = (re ** 2 + im ** 2 + eps ** 2) ** 0.5
        angle = torch.atan2(re, im)
        return mag, angle

    def inverse(self, re, im, transpose=False, squeeze=False):
        '''
            transpose (Boolean):
                [batch_size, num_frames, num_freq_bins] -> [batch_size, num_freq_bins, num_frames]
            squeeze (Boolean):
                [batch_size, 1, num_samples] -> [batch_size, num_samples]
        '''
        if transpose:
            re = re.transpose(1, 2)
            im = im.transpose(1, 2)

        # DC compensate
        if re.shape[1] == self.fft_size // 2:
            re = F.pad(re, (0, 0, 1, 0, 0, 0), 'constant')
            im = F.pad(im, (0, 0, 1, 0, 0, 0), 'constant')

        x = torch.cat([re, im], 1)
        x = F.conv_transpose1d(x,
                               Variable(self.inverse_basis, requires_grad=False),
                               stride=self.hop_size, padding=0)
        if squeeze:
            x = x.squeeze(1)
        return x

if __name__ == '__main__':
    import librosa
    from scipy.io import wavfile
    import matplotlib.pyplot as plt
    import time

    sample_rate, pcm = wavfile.read('/local/train_set/1100h_afe/eng100h_afe/5022-29411-0003.wav')
    #print(pcm.shape)
    pcm = pcm.astype(np.float32)
    #pcm = np.hstack([pcm for i in range(100)])

    sf = SogouSTFT(fft_size=512, hop_size=256)
    sf.cuda()
    #re, im = sf.transform(torch.from_numpy(pcm).unsqueeze(0), keep_dc=True, transpose=False)
    #print(re.shape, im.shape)

    pcm_tensor = torch.from_numpy(pcm).unsqueeze(0).cuda()
    start = time.clock()
    mag = sf.mag2(pcm_tensor, 160, keep_dc=True, transpose=False)
    cost = time.clock() - start
    print(cost)

    #print(mag.shape)
    mag = mag[0].cpu().detach().numpy()

    start = time.clock()
    #spec = librosa.stft(y=pcm, n_fft=512, hop_length=256)#.T
    spec = librosa.stft(y=pcm, n_fft=512, hop_length=160)#.T
    mag_librosa = np.abs(spec)
    cost = time.clock() - start
    print(cost)
    #print(spec.shape)

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.pcolormesh(np.log(mag_librosa + 0.1))
    plt.subplot(2, 1, 2)
    plt.pcolormesh(np.log(mag + 0.1))
    plt.savefig('./out.png')

