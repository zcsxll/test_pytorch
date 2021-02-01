import librosa
import soundfile
import numpy as np
import matplotlib.pyplot as plt

pcm, samplerate = soundfile.read('./test.wav')
print(pcm.shape)

ret = librosa.stft(pcm, n_fft=512, hop_length=160)
energy = np.sum(np.abs(ret) ** 2, axis = 0)
# print(energy.shape, energy[40:45])

plt.figure()
plt.subplot(4, 1, 1)
plt.plot(energy)
plt.subplot(4, 1, 2)
plt.plot(10 * np.log10(energy))
# plt.savefig('./out.png')

#pcm = pcm * 32767
dbs = []
powers = []
n_frames = (pcm.shape[0] - 512) // 160 + 1
print(n_frames)
for i in range(n_frames):
    frame = pcm[i * 160:i * 160 + 512]
    # frame = frame * 32767
    frame_power = np.sum(frame ** 2)
    powers.append(frame_power)
    dbs.append(10 * np.log10(frame_power))
# print(powers[41:46])

plt.subplot(4, 1, 3)
plt.plot(powers)
plt.subplot(4, 1, 4)
plt.plot(dbs)
plt.savefig('./out.png')
