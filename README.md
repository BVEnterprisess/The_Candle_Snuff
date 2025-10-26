<<<<<<< HEAD
# The_Candle_Snuff

This project sets up a fine-tuning environment using Hugging Face Transformers for micro agents.

## Setup

1. Install Python 3.11 or later.

2. Create a virtual environment:
    ```
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows: `venv\Scripts\activate`
    - On Linux/Mac: `source venv/bin/activate`

4. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

## Configuration

Edit `config.yaml` to set your model, training parameters, and data paths.

## Data Preparation

Place your CSV files in the `data/` directory:
- `train.csv`
- `val.csv`
- `test.csv`

Each CSV should have columns: `text`, `label`.

Run data preparation:
```
python data_prep.py
```

## Training

To fine-tune the model:
```
python train.py
```

## Evaluation

To evaluate the model:
```
python eval.py
```

## Hardware Requirements

Ensure you have sufficient GPU memory for the model. Check CUDA availability with:
```
python -c "import torch; print(torch.cuda.is_available())"
```

## Installing CUDA for GPU Support

To enable GPU acceleration, install CUDA:

1. Install NVIDIA drivers on Windows from [NVIDIA website](https://www.nvidia.com/drivers).

2. In WSL, install CUDA toolkit:
    ```
    wsl --exec wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    wsl --exec sudo dpkg -i cuda-keyring_1.1-1_all.deb
    wsl --exec sudo apt update
    wsl --exec sudo apt install cuda-toolkit-12-2
    ```

3. Add CUDA to PATH in WSL: Add to ~/.bashrc:
    ```
    export PATH="/usr/local/cuda-12.2/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"
    ```

4. Reinstall PyTorch with CUDA support:
    ```
    pip uninstall torch
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

5. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

## Colab Training

For Google Colab training, use the provided `web_scraping_colab.ipynb` notebook with the sample datasets in `data/`.

## Notes

- Model weights are saved in `models/fine_tuned_model/`.
- Logs are in `logs/`.
=======
# The_Candle_Snuff
>>>>>>> 43090c183bdb3d7b56f447bcf251dbbdc94873d0
