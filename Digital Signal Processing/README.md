# Speech Command Classification Project

This project implements a binary speech command classification system using a custom LogMelFilterBanks feature extraction class and a CNN model. The system is trained and evaluated on the Google Speech Commands dataset, specifically using only "YES" and "NO" commands.

Task description: https://github.com/AntonOkhotnikov/ai-talent-hub-itmo-speech-course/blob/main/assignments/assignment1/README.md

## Repository Structure

- `melbanks.py` - Implementation of the LogMelFilterBanks class
- `preprocessed_16khz_mono.wav` - Preprocessed audio sample at 16kHz
- `README.md` - This file
- `Report.docx` - Comprehensive report with methods and findings
- `Task_1_Evaluation` and `Task_2_Evaluation` - Task evaluation files

## Implementation Details

### Part 1: LogMelFilterBanks

The `melbanks.py` file contains the implementation of a custom LogMelFilterBanks class that transforms audio signals into log mel-filterbank features. This implementation:

- Uses PyTorch's STFT functionality
- Calculates power spectrum
- Applies mel-filterbank transformation
- Takes logarithm of the result

The implementation was validated against the native `torchaudio.transforms.MelSpectrogram` implementation.

### Part 2: CNN Model and Experiments

The project implements a CNN-based binary classifier using PyTorch Lightning. Experiments were conducted to analyze:

1. The effect of different numbers of mel-filterbanks (n_mels ∈ {20, 40, 80})
2. The effect of different values of the groups parameter in convolution layers (groups ∈ {1, 2, 4, 8})

## Key Findings

- Lower number of mel-filterbanks (n_mels=20) achieved the highest accuracy (99.39%)
- FLOPs increased proportionally with n_mels, while training time nearly doubled from n_mels=20 to n_mels=80
- Grouped convolutions offered a compelling trade-off between accuracy and computational efficiency
- Using groups=4 reduced FLOPs by approximately 3x with only a 0.13% drop in accuracy compared to groups=1

## Contacts

tg @wallrich
