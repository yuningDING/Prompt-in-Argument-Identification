# BEA-NAACL-2022-38
Experiment code of the paper "Don't Drop the Topic - The Role of the Prompt in Argument Identification in Student Writing" accepted by BEA-NAACL-2022.

### How to use

1. Download data from: https://www.kaggle.com/c/feedback-prize-2021/ and save data to './data'

2. Install environment

    ```bash
    conda create --name env python=3.7
    conda activate env
    pip install -r requirements.txt
    ```
    
3. Split data into clusters
    ```bash
    python ./datasplit.py
    ```

4. Training model with different settings for 15 prompts
    ```bash
    for PROMPT in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
    do
    python ./experiment_pipeline.py --train_prompt all_big --validate_prompt all_big --test_prompt ${PROMPT} --input ./BeaExperimentSplittedData --model allenai/longformer-large-4096 --lr 1e-5 --output ./output --max_len 1536 --epochs 10
    done
    ```
