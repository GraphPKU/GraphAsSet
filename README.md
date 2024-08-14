This repository contains the official code for the paper [Graph as Point Set](https://arxiv.org/pdf/2405.02795v2).

# Environment
We use python 3.10 with pytorch 2.0.1, torchmetrics 1.0.3, ogb 1.3.6, and pyg 2.3.1


# Reproduce our results

On ZINC:
```
cd zinc
python main.py --dataset zinc --repeat 10 --epochs 1200 --batch_size 96 --testbatch_size 96 --lr 0.0015 --warmstart 17 --conststep 1000 --cosstep 17 --lexp_layer 1 --decompnoise 1e-4  --gradclipnorm 1e-1 --wd 1e-1 --align_size 4
```

on ZINC-full:
```
cd zinc
mkdir mod

python main.py --dataset zinc-full --repeat 1 --epochs 360 --batch_size 128 --testbatch_size 128 --cosstep 40 --conststep 240 --warmstart 40 --num_layers 8 --decompnoise 0   --wd 1e-1 
```

qm9
```
cd qm9
target=1 # you can choose target from 0 to 11
python main.py --dataset qm9$target --repeat 1 --epochs 240 --batch_size 256 --testbatch_size 256 --warmstart 1  --cosstep 40 --conststep 199  --num_layers 8 --decompnoise 1e-5   --wd 1e-1 --use_y_scale
```

pascalvoc-sp
```
python main.py --dataset pascalvocsp --repeat 1 --epochs 40 --batch_size 6 --testbatch_size 6 --warmstart 5 --conststep 40 --cosstep 5 --num_layers 4 --decompnoise 1e-5   --wd 1e-1 --hiddim 96 --align_size 32
```

peptide-func
```
python main.py --dataset pepfunc --repeat 1 --epochs 80 --el_norm ln --batch_size 6 --testbatch_size 6 --lr 0.0003 --pool max --warmstart 8 --conststep 64 --cosstep 8 --noconv_tailact --nosv_tailact --novmean --novnorm --noelvmean --noelvnorm --align_size 32
```

peptide-struct
```
python main.py --dataset pepstruct --repeat 1 --epochs 80 --gsizenorm 1.9 --el_norm ln --batch_size 2 --testbatch_size 2 --lr 0.0008 --warmstart 80 --conststep 0 --cosstep 8 --num_layers 8 --l_layers 4 --decompnoise 1e-06 --beta 0.997 --noconv_tailact --nosv_tailact --novmean --novnorm --noelvmean --noelvnorm --align_size 32
```
