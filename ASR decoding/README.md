# ASR Decoding Methods Implementation

This repository contains implementation of various automatic speech recognition (ASR) decoding methods for the wav2vec2 CTC acoustic model.

## Project Structure

- `examples/` - Contains sample audio files for testing
- `lm/` - Contains KenLM language model files
- `ASR_decoding/` - Jupyter notebook with experiments and analysis
- `wav2vec2decoder.py` - Main implementation of the decoding methods
- `report.pdf` - Detailed report of implementation and experimental results
- `ASR_decoding.ipynb` - Experiments

## Implementation

The project implements four CTC decoding methods for the wav2vec2 ASR model:

1. **Greedy Decoding** - Selects the most probable token at each time step
2. **Beam Search Decoding** - Maintains multiple hypotheses to find a better path
3. **Beam Search with LM Fusion** - Incorporates language model during decoding
4. **Beam Search with LM Rescoring** - Applies language model to rescore hypotheses

## Usage

### Prerequisites

```bash
sudo sh -c 'apt-get update && apt-get upgrade && apt-get install cmake'
python3 -m pip install https://github.com/kpu/kenlm/archive/master.zip
python3 -m pip install levenshtein
python3 -m pip install torch torchaudio transformers