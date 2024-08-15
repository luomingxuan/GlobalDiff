
python -u GlobalDiff.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code ml-1m 
python -u GlobalDiff.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code kuaishou 
python -u GlobalDiff.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code beauty 

python -u GlobalDiff_bert.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code ml-1m 
python -u GlobalDiff_bert.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code kuaishou 
python -u GlobalDiff_bert.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code beauty 

python -u GlobalDiff_SasRec.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code ml-1m 
python -u GlobalDiff_SasRec.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code kuaishou 
python -u GlobalDiff_SasRec.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code beauty 

python -u Globaldiff_atten_adapter.py --timesteps 100 --lr 0.001 --optimizer adamw --diffuser_type Unet --random_seed 0 --dataset_code ml-1m