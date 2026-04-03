#!/bin/bash

ALLOCATE=false
HOURS=3

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --allocate) ALLOCATE=true ;;
        --hours) HOURS="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if $ALLOCATE; then
    if [ "$HOURS" -le 0 ]; then
        echo "Error: --hours must be > 0 when --allocate is set."
        exit 1
    fi

    echo "Requesting allocation for $HOURS hours..."
    echo "Please wait 2-3 minutes"
    # Example Slurm allocation:
    # salloc --time=${HOURS}:00:00
    # interact -p GPU-shared --gres=gpu:v100-32:1 -t "$HOURS:00:00" -A cis250019p

    # salloc -A cis250019p -p GPU-shared --gres=gpu:h100-80:1 -t "${HOURS}:00:00" \
    salloc -A cis260085p -p GPU-shared --gres=gpu:h100-80:1 -t "${HOURS}:00:00" \
        srun --pty bash -lc '
            module load anaconda3
            conda deactivate || true
            conda activate /ocean/projects/cis250019p/jli87/envs/IDLS26_clone
            export PYTHONNOUSERSITE=1
            echo "Launching Jupyter Notebook..."
            jupyter notebook --no-browser --ip=0.0.0.0
        '

else
    echo "Allocation flag not set."
fi


# remaining steps to set up the environment and launch Jupyter Notebook
module load anaconda3

conda deactivate
conda activate /ocean/projects/cis250019p/jli87/envs/IDLS26_clone && export PYTHONNOUSERSITE=1

echo "Launching Jupyter Notebook..."
jupyter notebook --no-browser --ip=0.0.0.0