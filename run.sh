#!/bin/bash

# Common argument values
EXTRA_SRC_PROMPT=", oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed"
EXTRA_TGT_PROMPT=", detailed high resolution, high quality, sharp"
DENOISE_CFG_SCALE=1
LR=0.01
DELTAT=80
DELTAS=200

# Define timesteps for each mode
declare -A TIMESTEPS
TIMESTEPS=(
    ["bridge"]=1000
    ["sds"]=1000
    ["vsd"]=1000
    ["lucid"]=1000
    ["jsdg"]=20000
)

# Seeds range
SEEDS=(0 1 2 3 4 5 6 7)

# Prompts
PROMPTS=("A cactus with pink flowers" "A white seashell on a sandy beach" "An artist is painting on a blank canvas")

# Loop over different prompts
for PROMPT in "${PROMPTS[@]}"
do
    # Loop over different modes
    for MODE in "${!TIMESTEPS[@]}"
    do
        N_STEPS=${TIMESTEPS[$MODE]}

        # Handle NUMT for jsdg
        if [ "$MODE" == "jsdg" ]; then
            NUMT_LIST=(1 2)  # Use both NUMT=1 and NUMT=2 for jsdg
        else
            NUMT_LIST=()  # Empty NUMT_LIST for other modes
        fi

        # Use different CFG_SCALE for vsd
        if [ "$MODE" == "vsd" ]; then
            CFG_SCALE_LIST=(7.5 100)  # Use both 7.5 and 100 for VSD
        else
            CFG_SCALE_LIST=(100)  # Default CFG_SCALE for other modes
        fi

        echo "Running mode: $MODE with n_steps: $N_STEPS - Prompt: $PROMPT"

        # Loop over CFG_SCALE values
        for CFG_SCALE in "${CFG_SCALE_LIST[@]}"
        do
            echo "  Using CFG_SCALE: $CFG_SCALE"

            # Loop over NUMT values (only for jsdg)
            if [ ${#NUMT_LIST[@]} -eq 0 ]; then
                # No NUMT for modes other than jsdg
                for SEED in "${SEEDS[@]}"
                do
                    echo "    Running with seed: $SEED"

                    python -Wignore main.py \
                        --prompt "$PROMPT" \
                        --extra_src_prompt "$EXTRA_SRC_PROMPT" \
                        --extra_tgt_prompt "$EXTRA_TGT_PROMPT" \
                        --mode "$MODE" \
                        --cfg_scale "$CFG_SCALE" \
                        --denoise_cfg_scale "$DENOISE_CFG_SCALE" \
                        --lr "$LR" \
                        --deltat "$DELTAT" \
                        --deltas "$DELTAS" \
                        --seed "$SEED" \
                        --n_steps "$N_STEPS"

                    echo "      Completed seed: $SEED for mode: $MODE"
                done
            else
                # Handle NUMT for jsdg
                for NUMT in "${NUMT_LIST[@]}"
                do
                    echo "    Using NUMT: $NUMT"

                    for SEED in "${SEEDS[@]}"
                    do
                        echo "      Running with seed: $SEED"

                        python -Wignore main.py \
                            --prompt "$PROMPT" \
                            --extra_src_prompt "$EXTRA_SRC_PROMPT" \
                            --extra_tgt_prompt "$EXTRA_TGT_PROMPT" \
                            --mode "$MODE" \
                            --cfg_scale "$CFG_SCALE" \
                            --denoise_cfg_scale "$DENOISE_CFG_SCALE" \
                            --lr "$LR" \
                            --deltat "$DELTAT" \
                            --deltas "$DELTAS" \
                            --numt "$NUMT" \
                            --seed "$SEED" \
                            --n_steps "$N_STEPS"

                        echo "        Completed seed: $SEED with NUMT: $NUMT and CFG_SCALE: $CFG_SCALE for mode: $MODE"
                    done
                done
            fi
        done

        echo "Completed mode: $MODE with n_steps: $N_STEPS"
        echo "---------------------------------"
    done
done