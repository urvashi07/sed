conda create -y -n sed_new python==3.8.5
conda activate sed_new
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y librosa ffmpeg sox pandas numba scipy torchmetrics youtube-dl tqdm pytorch-lightning=1.9 -c conda-forge
pip install tensorboard
pip install h5py
pip install thop
pip install codecarbon==2.1.4
pip install -r requirements.txt
pip install museval
pip install torchlibrosa
pip install -e ../../.
pip install torchcontrib==0.0.2
pip install audiomentations
