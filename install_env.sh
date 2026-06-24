conda create --name universal_vlm_test python=3.8.20
conda activate universal_vlm_test

pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
