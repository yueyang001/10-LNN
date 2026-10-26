"""读取真实 ShipsEar 音频，生成并保存 3 秒 16 kHz Log-Mel 频谱图。"""

import argparse
import math
import time
from pathlib import Path

import matplotlib
import torch
import torchaudio


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAMPLE_RATE = 16_000
DURATION_SECONDS = 3.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 64
DATA_DIR = Path("/media/hdd1/chuxiaohui/AI4Ocean_UATR/ShipsEar_622")
OUTPUT_DIR = Path("/media/hdd1/fubohan/Code/UATR/docs")


def find_audio_file(data_dir):
    """在数据集目录中选择排序后的第一段真实 WAV 音频。"""
    audio_paths = sorted(data_dir.rglob("*.wav"))
    if not audio_paths:
        raise FileNotFoundError("未在数据集目录中找到 .wav 音频: {}".format(data_dir))
    return audio_paths[0]


def load_audio(audio_path):
    """读取原始音频，并返回读取耗时（毫秒）。"""
    start_time = time.perf_counter()
    waveform, source_sample_rate = torchaudio.load(str(audio_path))
    elapsed_ms = (time.perf_counter() - start_time) * 1_000.0
    return waveform, source_sample_rate, elapsed_ms


def prepare_audio(waveform, source_sample_rate, device):
    """转换为单声道、16 kHz 且严格为 3 秒的音频输入。"""
    start_time = time.perf_counter()

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if source_sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=source_sample_rate,
            new_freq=SAMPLE_RATE,
        )
        waveform = resampler(waveform)

    if waveform.shape[-1] >= NUM_SAMPLES:
        waveform = waveform[:, :NUM_SAMPLES]
    else:
        waveform = torch.nn.functional.pad(
            waveform,
            (0, NUM_SAMPLES - waveform.shape[-1]),
        )

    waveform = waveform.to(device)
    elapsed_ms = (time.perf_counter() - start_time) * 1_000.0
    return waveform, elapsed_ms


def hz_to_mel(frequency):
    return 2595.0 * math.log10(1.0 + frequency / 700.0)


def mel_to_hz(mel):
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def create_mel_filterbank(device):
    """创建形状为 [N_MELS, N_FFT // 2 + 1] 的 Mel 滤波器组。"""
    mel_points = torch.linspace(
        hz_to_mel(0.0),
        hz_to_mel(SAMPLE_RATE / 2.0),
        N_MELS + 2,
        device=device,
        dtype=torch.float32,
    )
    hz_points = mel_to_hz(mel_points)
    fft_frequencies = torch.linspace(
        0.0,
        SAMPLE_RATE / 2.0,
        N_FFT // 2 + 1,
        device=device,
        dtype=torch.float32,
    )
    lower_slope = (
        fft_frequencies.unsqueeze(0) - hz_points[:-2].unsqueeze(1)
    ) / (hz_points[1:-1] - hz_points[:-2]).unsqueeze(1)
    upper_slope = (
        hz_points[2:].unsqueeze(1) - fft_frequencies.unsqueeze(0)
    ) / (hz_points[2:] - hz_points[1:-1]).unsqueeze(1)
    return torch.clamp(torch.minimum(lower_slope, upper_slope), min=0.0)


def waveform_to_log_mel(waveform, window, mel_filterbank):
    """将单声道波形转换为 Log-Mel 功率谱图。"""
    spectrum = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=window,
        center=True,
        return_complex=True,
    )
    power_spectrogram = spectrum.abs().pow(2.0)
    mel_spectrogram = torch.matmul(mel_filterbank, power_spectrogram)
    return 10.0 * torch.log10(torch.clamp(mel_spectrogram, min=1e-10))


def synchronize(device):
    """确保 CUDA 异步任务执行完成后再结束计时。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def convert_to_log_mel(waveform, device):
    """生成 Log-Mel 频谱图，并返回转换耗时（毫秒）。"""
    window = torch.hann_window(N_FFT, device=device)
    mel_filterbank = create_mel_filterbank(device)

    synchronize(device)
    start_time = time.perf_counter()
    with torch.no_grad():
        log_mel = waveform_to_log_mel(waveform, window, mel_filterbank)
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start_time) * 1_000.0
    return log_mel, elapsed_ms


def save_mel_image(log_mel, output_path):
    """保存 PNG 格式的 Log-Mel 频谱图，并返回保存耗时（毫秒）。"""
    start_time = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mel_image = log_mel.squeeze(0).detach().cpu().numpy()
    plt.imsave(str(output_path), mel_image, cmap="magma", origin="lower")
    elapsed_ms = (time.perf_counter() - start_time) * 1_000.0
    return elapsed_ms


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="ShipsEar 数据集根目录",
    )
    parser.add_argument(
        "--audio-path",
        type=Path,
        default=None,
        help="指定待处理的 WAV 文件；未指定时自动选择数据集中的第一段音频",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Mel 频谱图的输出目录",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Mel 频谱图计算设备，默认使用 CPU",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境没有可用的 CUDA 设备")
    if args.audio_path is not None and not args.audio_path.is_file():
        parser.error("指定的音频文件不存在: {}".format(args.audio_path))
    if args.audio_path is None and not args.data_dir.is_dir():
        parser.error("数据集目录不存在: {}".format(args.data_dir))
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    audio_path = args.audio_path or find_audio_file(args.data_dir)
    output_path = args.output_dir / "{}_mel.png".format(audio_path.stem)

    waveform, source_sample_rate, read_time_ms = load_audio(audio_path)
    waveform, preprocess_time_ms = prepare_audio(
        waveform,
        source_sample_rate,
        device,
    )
    log_mel, conversion_time_ms = convert_to_log_mel(waveform, device)
    save_time_ms = save_mel_image(log_mel, output_path)

    print("音频文件: {}".format(audio_path))
    print("设备: {}".format(device))
    print("原始采样率: {} Hz".format(source_sample_rate))
    print("Mel 输入: {:.1f} 秒, {} Hz, {} 个采样点".format(
        DURATION_SECONDS,
        SAMPLE_RATE,
        waveform.shape[-1],
    ))
    print("Log-Mel 形状: {}".format(tuple(log_mel.shape)))
    print("读取音频耗时: {:.3f} ms".format(read_time_ms))
    print("音频预处理耗时: {:.3f} ms".format(preprocess_time_ms))
    print("转换 Log-Mel 耗时: {:.3f} ms".format(conversion_time_ms))
    print("保存 Mel 频谱图耗时: {:.3f} ms".format(save_time_ms))
    print("总耗时: {:.3f} ms".format(
        read_time_ms + preprocess_time_ms + conversion_time_ms + save_time_ms
    ))
    print("Mel 频谱图已保存至: {}".format(output_path))


if __name__ == "__main__":
    main()
