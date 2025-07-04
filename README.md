# Reflective Surfaces (RS)

## Installation

### Manual

#### OS

Ensure you have `Ubuntu 22.04` installed.

The home directory is structured as follows:

```bash
./home
├── .bashrc
├── .config
├── research
```

#### Virtual Environment

Ensure you have `Python 3.10-3.11` installed.

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Blender

get the blender from Google Drive

```bash
cd home
pip install gdown
gdown --folder https://drive.google.com/drive/u/1/folders/1sHqz5PRKtLQI0aEcByzKMyNwIOSG557l
```

There are two zip files, one for the blender/saved models and one for the blender config. Unzip them and put them in the root directory `home` of the system.

```bash
cd blender_gdown
unzip blender.zip
mv blender home
unzip blender_config.zip
mv blender home/.cache
cd ..
rm -rf blender_gdown
```

Now, the home directory should look like this:

```bash
./home
├── .bashrc
├── .cache
│   └── blender
├── blender
│   ├── addons
│   ├── blender-3.3.14-linux-x64
│   └── models
├── research
```

#### Dependencies

Ensure you have NVIDIA drivers installed.

```bash
cd home/research
git clone https://github.com/hieutrungle/rs
cd rs
pip install -e .
pip install -r requirements.txt
```

## Usage

To train:

```bash
cd home/research/rs
bash run.sh
```

### Ablation

- No compatibility scores: `--no_compatibility_scores True`
- Random assignment of users to targets: `--random_assignment True`
- No allocator, simple MARL with the shared reward is the sum of all RSSI: `--no_allocator True`

### Video Generation

Remember to create a directory for the videos: `mkdir {OUTPUT_VIDEO_DIR}`

```bash
ffmpeg -framerate 5 -i {PATH_TO_IMAGES}_%05d.png -r 30 -pix_fmt yuv420p {OUTPUT_VIDEO_PATH}.mp4
```

Example:

```bash
ffmpeg -framerate 5 -i ./tmp_long_short_mean_adjusted_local_assets/images/SAC_Mean_Adjusted__orin__wireless-sigmap-v0__fecc18e6_03-12-2024_17-09-34_0/hallway_L_0_%05d.png -r 30 -pix_fmt yuv420p ./tmp_long_short_mean_adjusted_local_assets/videos/SAC_Mean_Adjusted__orin__wireless-sigmap-v0__fecc18e6_03-12-2024_17-09-34_0.mp4
```

## Completed Tasks

```markdown
[x] Torchrl environment
[x] Torchrl environment with ParallelEnv 
```

```markdown
Note: Comment out daemon in ParallelEnv to avoid issues with Simulation Worker, which is a child process to overcome memory leak of continuous Sionna scene loading.
The commented line is:
# process.daemon = True
rs_venv/lib/python3.11/site-packages/torchrl/envs/batched_envs.py
```
