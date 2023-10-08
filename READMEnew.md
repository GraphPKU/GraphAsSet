# Environment
We use python 3.10 with pytorch 2.0.1, torchmetrics 1.0.3, ogb 1.3.6, and pyg 2.3.1


# Reproduce our results

On ZINC:
```
cd zinc
python main.py --dataset zinc --repeat 10 --epochs 493 --batch_size 96 --testbatch_size 96 --lr 0.0015 --warmstart 17 --conststep 313 --cosstep 17 --lexp_layer 1 --decompnoise 1e-4  --gradclipnorm 1e-1 --wd 1e-1 
```

on ZINC-full:
```
cd zinc
mkdir mod

python main.py --dataset zinc-full --repeat 1 --epochs 360 --batch_size 128 --testbatch_size 128 --cosstep 40 --constep 240 --warmstart 40 --num_layers 8 --decompnoise 0   --wd 1e-1 
```

qm9
```
cd qm9
target=1 # you can choose target from 0 to 11
python main.py --dataset qm9$target --repeat 1 --epochs 140 --batch_size 256 --testbatch_size 256 --warmstart 1  --cosstep 40 --cosstep 99  --num_layers 8 --decompnoise 1e-5   --wd 1e-1 --use_y_scale
```

pascalvoc-sp
```
python main.py --dataset pascalvocsp --repeat 1 --epochs 40 --batch_size 16 --testbatch_size 16 --warmstart 0 --conststep 100 --num_layers 8 --decompnoise 1e-5   --wd 1e-1 
```

peptide-func
```
cd pep
python main.py --dataset pepfunc --repeat 1 --epochs 80 --gsizenorm 1.9 --el_norm ln --batch_size 2 --testbatch_size 2 --lr 0.0008 --pool max --warmstart 40 --conststep 0 --cosstep 20 --l_layers 4 --minlr 4e-5 --decompnoise 1e-06 --beta 0.997 --noconv_tailact --nosv_tailact --novmean --novorm --noelvmean --noelvnorm
```

peptide-struct
```
cd pep
python main.py --dataset pepstruct --repeat 1 --epochs 40 --gsizenorm 1.9 --el_norm ln --batch_size 2 --testbatch_size 2 --lr 0.0008 --K 0.1 --warmstart 40 --conststep 0 --cosstep 8 --num_layers 8 --l_layers 4 --decompnoise 1e-06 --beta 0.997 --noconv_tailact --nosv_tailact --novmean --novorm --noelvmean --noelvnorm
```