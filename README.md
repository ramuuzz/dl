# Deep Learning Experiments — Simplified and Consistent

These files rewrite the experiments from the supplied repository with a consistent order:

1. Imports
2. Device (when using PyTorch)
3. Dataset loading and preprocessing
4. DataLoader / sequence preparation
5. Model creation
6. Loss and optimizer
7. Training
8. Testing / prediction / plots

## Files
- `mp.py` — manual neural-network forward calculation
- `linear.py` — linear regression from a CSV
- `feed_forwrd.py` — feedforward neural network on CIFAR-10
- `cnn.py` — CNN on MNIST
- `optim.py` — initialization, dropout and weight-decay comparison on MNIST
- `6t.py` — transfer learning with pretrained VGG19 on MNIST
- `50.py`, `nifty.py` — NIFTY-50 time-series forecasting with LSTM (same experiment)
- `7t.py` — IMDB sentiment classification with LSTM
- `8t.py` — comparison of RNN, GRU and LSTM on IMDB
- `eng.py` — English-to-Hindi encoder-decoder GRU
- `machine_translation.py` — TF-IDF + small neural retrieval/translation demo
- `image_en.py` — image enhancement, grayscale and thresholding

## Install common dependencies
```bash
pip install torch torchvision matplotlib pandas numpy scikit-learn pillow tensorflow yfinance
```

Not every experiment needs every package. VGG19 downloads pretrained weights; MNIST/CIFAR-10 and IMDB may download datasets. Stock forecasting requires internet access and `yfinance`.

## Dataset files you must supply
- `linear.py`: a CSV named `data.csv` with columns `X` and `Y` in the same folder.
- `eng.py`: a CSV named `Dataset.csv` with columns `English` and `Hindi` in the same folder.
- `machine_translation.py`: a CSV named `Dataset_English_Hindi.csv` with columns `English` and `Hindi` in the same folder.
- `image_en.py`: an image named `image.png` in the same folder.

## Notes
- These scripts are teaching-friendly rewrites, not byte-for-byte copies. Hard-coded personal computer paths have been replaced with relative paths.
- Epoch counts and dataset sizes are kept small where practical, but CNN, VGG19, recurrent models and translation can still take time.
- `nn.Sequential` is used where layers can be chained directly. Custom `nn.Module` classes remain for LSTM/GRU models where the hidden-state tuple needs to be handled.
- `machine_translation.py` retrieves the closest existing English sentence and its paired Hindi sentence; it is not a full generative translation system.
