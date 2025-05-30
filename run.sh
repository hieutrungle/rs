#!/usr/bin/env bash

set -o pipefail # Pipe fails when any command in the pipe fails
set -u  # Treat unset variables as an error

handle_error() {
    echo "An error occurred on line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done

# # Source: https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
# # Get the directory of the script (does not solve symlink problem)
# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# echo "Script directory: $SCRIPT_DIR"

# Get the source path of the script, even if it's called from a symlink
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPT_DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
echo "Source directory: $SCRIPT_DIR"
SOURCE_DIR=$SCRIPT_DIR
ASSETS_DIR=${SOURCE_DIR}/local_assets

# * Change this to your blender directory
RESEARCH_DIR=$(dirname $SOURCE_DIR)
HOME_DIR=$(dirname $RESEARCH_DIR)
BLENDER_DIR=${HOME_DIR}/blender 

echo Blender directory: $BLENDER_DIR
echo Coverage map directory: $SOURCE_DIR
echo -e Assets directory: $ASSETS_DIR '\n'

python ./main.py --command "train" --env_id "classroom" --checkpoint_dir "/home/hieule/research/rs/local_assets/models" --sionna_config_file "/home/hieule/research/rs/configs/sionna_shared_ap.yaml" --replay_buffer_dir "/home/hieule/research/rs/local_assets/replay_buffer" --source_dir "/home/hieule/research/rs/" --num_envs 5 
# --ep_len 3 --frames_per_batch 5 --n_iters 2 --num_epochs 2 --minibatch_size 2 --wandb "offline" 
# --load_model "/home/hieule/research/rs/local_assets_2/models/checkpoint_1.pt"
# python ./main.py --command "eval" --checkpoint_dir "/home/hieule/research/rs/local_assets/models" --sionna_config_file "/home/hieule/research/rs/configs/sionna_shared_ap.yaml" --replay_buffer_dir "/home/hieule/research/rs/local_assets/replay_buffer" --wandb "offline" --num_envs 1