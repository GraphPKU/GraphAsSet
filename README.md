# Environment
We use python 3.10 with pytorch 2.0.1, torchmetrics 1.0.3, ogb 1.3.6, and pyg 2.3.1


# Reproduce our results

On ZINC:
```
cd zinc
python main.py --dataset zinc --repeat 10 --epochs 493  --elres --usesvmix --vmean --vnorm --elvmean --elvnorm --snorm --gsizenorm 1.85 --lsizenorm 0.2 --el_numlayer 2   --el_tailact --conv_numlayer 1 --conv_norm none --conv_uselinv --conv_tailact --dppreepoch 0 --batch_size 96 --testbatch_size 96 --dp 0.0 --lr 0.0015 --pool mean --K 0.0 --K2 0 --warmstart 17 --lossparam 0.0 --act silu --featdim 16 --hiddim 128 --caldim 128 --num_layers 6 --l_encoder deepset --l_layers 3 --l_aggr mean --l_mlplayers1 2 --l_mlplayers2 0 --l_mlpnorm1 ln --l_mlpnorm2 none --l_res --l_mlptailact1 --sv_numlayer 1 --sv_norm none --sv_uselinv --sv_tailact --sv_res --predlin_numlayer 1 --predlin_norm none --lexp_layer 1 --lexp_norm ln --minlr 0 --seedoffset 0 --decompnoise 1e-4 --gradclipnorm 1e-1 --wd 1e-1 --embln
```

on ZINC-full:
```
cd zinc
mkdir mod

python main.py --dataset zinc-full --repeat 1 --epochs 360  --elres --usesvmix --vmean --vnorm --elvmean --elvnorm --snorm --gsizenorm 1.85 --lsizenorm 0.2 --el_numlayer 2   --el_tailact --conv_numlayer 1 --conv_norm none --conv_uselinv --conv_tailact --dppreepoch 0 --batch_size 128 --testbatch_size 128 --dp 0.0 --lr 0.001 --pool mean --K 0.0 --K2 0 --warmstart 40 --lossparam 0.0 --act silu --featdim 16 --hiddim 128 --caldim 128 --num_layers 8 --l_encoder deepset --l_layers 3 --l_aggr mean --l_mlplayers1 2 --l_mlplayers2 0 --l_mlpnorm1 ln --l_mlpnorm2 none --l_res --l_mlptailact1 --sv_numlayer 1 --sv_norm none --sv_uselinv --sv_tailact --sv_res --predlin_numlayer 1 --predlin_norm none --lexp_layer 2 --lexp_norm ln --minlr 0 --seedoffset 7 --decompnoise 0 --gradclipnorm 1 --wd 1e-1 --embln 
```

qm9
```
cd qm9
target=1 # you can choose target from 0 to 11
python main.py --dataset qm9$target --repeat 1 --epochs 140  --elres --usesvmix --vmean --vnorm --elvmean --elvnorm --snorm --gsizenorm 1.85 --lsizenorm 0.2 --el_numlayer 2   --el_tailact --conv_numlayer 1 --conv_norm none --conv_uselinv --conv_tailact --dppreepoch 0 --batch_size 256 --testbatch_size 256 --dp 0.0 --lr 0.001 --pool mean --K 0.0 --K2 0 --warmstart 40 --lossparam 0.0 --act silu --featdim 128 --hiddim 128 --caldim 128 --num_layers 8 --l_encoder deepset --l_layers 3 --l_aggr mean --l_mlplayers1 2 --l_mlplayers2 0 --l_mlpnorm1 ln --l_mlpnorm2 none --l_res --l_mlptailact1 --sv_numlayer 1 --sv_norm none --sv_uselinv --sv_tailact --sv_res --predlin_numlayer 1 --predlin_norm none --lexp_layer 2 --lexp_norm ln --minlr 0 --seedoffset 0 --decompnoise 1e-5 --gradclipnorm 1 --wd 1e-1 --embln 
```

ogbg-molhiv
```
cd ogb
python main.py --dataset ogbg-molhiv --repeat 1 --epochs 300  --elres --usesvmix --vmean --vnorm --elvmean --elvnorm --snorm --gsizenorm 1.85 --lsizenorm 0.2 --el_numlayer 2   --el_tailact --conv_numlayer 1 --conv_norm none --conv_uselinv --conv_tailact --dppreepoch 0 --batch_size 24 --testbatch_size 24 --dp 0.0 --lr 0.001 --pool mean --K 0.0 --K2 0 --warmstart 40 --lossparam 0.0 --act silu --featdim 96 --hiddim 96 --caldim 96 --num_layers 6 --l_encoder deepset --l_layers 3 --l_aggr mean --l_mlplayers1 2 --l_mlplayers2 0 --l_mlpnorm1 ln --l_mlpnorm2 none --l_res --l_mlptailact1 --sv_numlayer 1 --sv_norm none --sv_uselinv --sv_tailact --sv_res --predlin_numlayer 1 --predlin_norm none --lexp_layer 2 --lexp_norm ln --minlr 0 --seedoffset 0 --decompnoise 1e-5 --gradclipnorm 1 --wd 1e-1 --embln
```

pascalvoc-sp
```
python main.py --dataset pascalvocsp --repeat 1 --epochs 40  --elres --usesvmix --vmean --vnorm --elvmean --elvnorm --snorm --gsizenorm 1.85 --lsizenorm 0.2 --el_numlayer 2   --el_tailact --conv_numlayer 1 --conv_norm none --conv_uselinv --conv_tailact --dppreepoch 0 --batch_size 16 --testbatch_size 16 --dp 0.0 --lr 0.001 --pool mean --K 0.0 --K2 0 --warmstart 0 --conststep 100 --lossparam 0.0 --act silu --featdim 128 --hiddim 128 --caldim 128 --num_layers 8 --l_encoder deepset --l_layers 3 --l_aggr mean --l_mlplayers1 2 --l_mlplayers2 0 --l_mlpnorm1 ln --l_mlpnorm2 none --l_res --l_mlptailact1 --sv_numlayer 1 --sv_norm none --sv_uselinv --sv_tailact --sv_res --predlin_numlayer 1 --predlin_norm none --lexp_layer 2 --lexp_norm ln --minlr 0 --seedoffset 0 --decompnoise 1e-5 --gradclipnorm 1 --wd 1e-1 --embln
```

peptide-func
```
cd pep
python main.py --dataset pepfunc --repeat 1 --epochs 80  --usesvmix --gsizenorm 1.9 --lsizenorm 0.4 --el_numlayer 2 --el_norm ln   --conv_numlayer 1 --conv_norm none --conv_uselinv  --dppreepoch 0 --batch_size 2 --testbatch_size 2 --dp 0.0 --lr 0.0008 --pool max --K 0.0 --K2 0 --warmstart 40 --conststep 0 --cosstep 20 --lossparam 0.0 --act silu --featdim 128 --hiddim 128 --caldim 128 --num_layers 6 --l_encoder deepset --l_layers 4 --l_aggr mean --l_mlplayers1 2 --l_mlplayers2 0 --l_mlpnorm1 ln --l_mlpnorm2 none --l_res --l_mlptailact1 --sv_numlayer 1 --sv_norm none --sv_uselinv --predlin_numlayer 1 --predlin_norm none --lexp_layer 2 --lexp_norm ln --minlr 4e-5 --decompnoise 1e-06 --gradclipnorm 1 --wd 0 --embln --beta 0.997  --elres  --el_tailact   --snorm  --sv_res
```

peptide-struct
```
cd pep
python main.py --dataset pepstruct --repeat 1 --epochs 40  --usesvmix --gsizenorm 1.9 --lsizenorm 0.4 --el_numlayer 2 --el_norm ln   --conv_numlayer 1 --conv_norm none --conv_uselinv  --dppreepoch 0 --batch_size 2 --testbatch_size 2 --dp 0.0 --lr 0.0008 --pool mean --K 0.1 --K2 0 --warmstart 40 --conststep 0 --cosstep 8 --lossparam 0.0 --act silu --featdim 128 --hiddim 128 --caldim 128 --num_layers 8 --l_encoder deepset --l_layers 4 --l_aggr mean --l_mlplayers1 2 --l_mlplayers2 0 --l_mlpnorm1 ln --l_mlpnorm2 none --l_res --l_mlptailact1 --sv_numlayer 1 --sv_norm none --sv_uselinv --predlin_numlayer 1 --predlin_norm none --lexp_layer 2 --lexp_norm ln --minlr 0 --decompnoise 1e-06 --gradclipnorm 1 --wd 0 --embln --beta 0.997  --elres  --el_tailact   --snorm  --sv_res
```