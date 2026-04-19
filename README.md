## Requirement
Checking and installing environmental requirements
```python
pip install -r requirements.txt
```
## Datasets

Google Drive: https://drive.google.com/drive/folders/1l_ex1wnAAMpEtO71rjjM1MKC7W_olEVi?usp=drive_link

Adding the dataset path to the corresponding location in the run.py file, e.g. IEMOCAP_path = "".

## Run
### IEMOCAP-6
```bash
python -u run.py --gpu 0 --port 1530 --classify emotion \
--dataset IEMOCAP --epochs 120 --textf_mode textf0 \
--loss_type emo_sen_sft --lr 1e-04 --batch_size 16 --hidden_dim 512 \
--win 17 17 --heter_n_layers 7 7 7 --drop 0.2 --shift_win 19 --lambd 1.0 1.0 0.7
```

### IEMOCAP-4
```bash
python -u run.py --gpu 0 --port 1531 --classify emotion \
--dataset IEMOCAP4 --epochs 120 --textf_mode textf0 \
--loss_type emo_sen_sft --lr 3e-04 --batch_size 16 --hidden_dim 256 \
--win 5 5 --heter_n_layers 4 4 4 --drop 0.2 --shift_win 10 --lambd 1.0 0.6 0.6
```

### MELD
```bash
python -u run.py --gpu 0 --port 1532 --classify emotion \
--dataset MELD --epochs 50 --textf_mode textf0 \
--loss_type emo_sen_sft --lr 7e-05 --batch_size 16 --hidden_dim 384 \
--win 3 3 --heter_n_layers 5 5 5 --drop 0.2 --shift_win 3 --lambd 1.0 0.5 0.2
```

