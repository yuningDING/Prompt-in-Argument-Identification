# BEA-NAACL-2022-38
Experiment code of the paper "Don't Drop the Topic - The Role of the Prompt in Argument Identification in Student Writing" accepted by BEA-NAACL-2022.

### How to use

1. Download data from: https://www.kaggle.com/c/feedback-prize-2021/ and save data to './data'

2. Install environment

    ```bash
    conda create --name env python=3.7
    conda activate env
    pip install -r experiment_requirements.txt
    ```
    
3. Split data into clusters and different experiment settings
    ```bash
    python ./data_split.py
    ```

4. Training model with different settings for 15 prompts.
For example, the same_prompt setting will be trained by:
    ```bash
    for PROMPT in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
    do
    python ./experiment_pipeline.py --train_prompt ${PROMPT} --validate_prompt ${PROMPT} --test_prompt ${PROMPT} --input ./data/same_prompt --model allenai/longformer-large-4096 --lr 1e-5 --output ./output --max_len 1536 --epochs 10
    done
    ```


### How to cite
   ```
   @inproceedings{ding-etal-2022-dont,
    title = "Don{'}t Drop the Topic - The Role of the Prompt in Argument Identification in Student Writing",
    author = "Ding, Yuning  and
      Bexte, Marie  and
      Horbach, Andrea",
    booktitle = "Proceedings of the 17th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2022)",
    month = jul,
    year = "2022",
    address = "Seattle, Washington",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.bea-1.17",
    doi = "10.18653/v1/2022.bea-1.17",
    pages = "124--133",
    }
