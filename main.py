from sklearn.metrics import roc_auc_score
from typing import Callable
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import argparse
import time
import numpy as np
from datasets import loaddataset
from SizeAlignedLoader import batch2dense
import numpy as np
from PiOModel import PiOModel
import os
from LRScheduler import CosineAnnealingWarmRestarts

def get_criterion(task, args):
    if task == "smoothl1reg":
        return torch.nn.SmoothL1Loss(reduction="none", beta=args.lossparam)
    else:
        criterion_dict = {
            "bincls": torch.nn.BCEWithLogitsLoss(reduction="none"),
            "cls": torch.nn.CrossEntropyLoss(reduction="none"),
            "reg": torch.nn.MSELoss(reduction="none"),
            "l1reg": torch.nn.L1Loss(reduction="none"),
        }
        return criterion_dict[task]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train(criterion: Callable,
          model: PiOModel,
          device: torch.device,
          loader: DataLoader,
          optimizer: optim.Optimizer,
          task_type: str,
          ampscaler: torch.cuda.amp.GradScaler = None,
          advloss: bool=False,
          scheduler: optim.lr_scheduler.LinearLR = None,
          gradclipnorm: float=1,
          args=None):
    model.train()
    losss = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', enabled=ampscaler is not None):
            datatuple = batch2dense(batch, aligned_size=args.align_size)[:-1]
            finalpred = model(*datatuple)
            finalpred = finalpred * model.ystd + model.ymean
            y = batch.y
            if task_type != "cls":
                y = y.to(torch.float)
            if advloss:
                value_loss = criterion(finalpred, y).flatten()
                weight = torch.softmax(value_loss.detach(), dim=-1)
                value_loss = torch.inner(value_loss, weight)
            else:
                value_loss = torch.mean(criterion(finalpred, y))
        if ampscaler is not None:
            ampscaler.scale(value_loss).backward()
            ampscaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradclipnorm)
            ampscaler.step(optimizer)
            ampscaler.update()
        else:
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradclipnorm)
            optimizer.step()
        scheduler.step()
        losss.append(value_loss)
    loss = np.average([_.item() for _ in losss])
    return loss

@torch.no_grad()
def eval(model: PiOModel,
         device: torch.device,
         loader: DataLoader,
         evaluator,
         amp: bool = False,
         args = None):
    model.eval()
    ty = loader.dataset.y
    ylen = ty.shape[0]
    if ty.dtype == torch.long:
        y_true = torch.zeros((ylen), dtype=ty.dtype)
    else:
        y_true = torch.zeros((ylen, model.num_tasks), dtype=ty.dtype)
    y_pred = torch.zeros((ylen, model.num_tasks), device=device)
    step = 0
    for batch in loader:
        steplen = batch.y.shape[0]
        y_true[step:step + steplen] = batch.y
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', enabled=amp):
            datatuple = batch2dense(batch, aligned_size=args.align_size)[:-1]
            tpred = model(*datatuple)
            tpred = tpred * model.ystd + model.ymean
        y_pred[step:step + steplen] = tpred
        step += steplen
    assert step == y_true.shape[0]
    if torch.any(torch.isnan(y_pred)):
        return float("nan")
    y_pred = y_pred.cpu()
    return evaluator(y_pred, y_true)


def parserarg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv")
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument("--noamp", dest="amp", action="store_false")
    parser.add_argument("--nocompile", dest="compile", action="store_false")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--testbatch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--minlr', type=float, default=0.0)
    parser.add_argument('--K', type=float, default=0.0)

    parser.add_argument('--gradclipnorm', type=float, default=1.0)
    parser.add_argument('--decompnoise', type=float, default=0)

    parser.add_argument('--seedoffset', type=int, default=0)

    parser.add_argument('--warmstart', type=int, default=0)
    parser.add_argument('--conststep', type=int, default=100)
    parser.add_argument('--cosstep', type=int, default=40)
    parser.add_argument('--use_y_scale', action="store_true")

    parser.add_argument('--dp', type=float, default=0.0)
    parser.add_argument('--eldp', type=float, default=0.0)

    parser.add_argument("--act", type=str, default="silu")

    parser.add_argument('--lossparam', type=float, default=0.0)
    parser.add_argument('--advloss', action="store_true")

    parser.add_argument('--embdp', type=float, default=0.0)
    parser.add_argument("--embbn", action="store_true")
    parser.add_argument("--emborthoinit", action="store_true")
    parser.add_argument("--degreeemb", action="store_true")
    parser.add_argument("--noembln", dest="embln", action="store_false")
    
    parser.add_argument('--featdim', type=int, default=-1)
    parser.add_argument('--hiddim', type=int, default=128)
    parser.add_argument('--caldim', type=int, default=-1)

    parser.add_argument("--normA", action="store_true")
    parser.add_argument('--nolaplacian', dest="laplacian", action="store_false")
    parser.add_argument('--nosqrtlambda', dest="sqrtlambda", action="store_false")
    
    parser.add_argument("--noelres", dest="elres", action="store_false")
    parser.add_argument("--nousesvmix", dest="usesvmix", action="store_false")
    parser.add_argument("--novmean", dest="vmean", action="store_false")
    parser.add_argument("--novnorm", dest="vnorm", action="store_false")
    parser.add_argument("--noelvmean", dest="elvmean", action="store_false")
    parser.add_argument("--noelvnorm", dest="elvnorm", action="store_false")
    parser.add_argument("--nosnorm", dest="snorm", action="store_false")

    parser.add_argument("--gsizenorm", type=float, default=1.85)
    
    parser.add_argument("--l_encoder", type=str, default="deepset")
    parser.add_argument("--l_layers", type=int, default=3)
    parser.add_argument("--l_combine", type=str, default="mul")
    parser.add_argument('--l_aggr', type=str, default="mean")
    parser.add_argument("--nol_res", dest="l_res", action="store_false")
    parser.add_argument("--nol_mlptailact1", dest="l_mlptailact1", action="store_false")
    parser.add_argument("--l_mlplayers1", type=int, default=2)
    parser.add_argument("--l_mlpnorm1", type=str, default="ln")
    parser.add_argument("--l_mlptailact2", action="store_true")
    parser.add_argument("--l_mlplayers2", type=int, default=0)
    parser.add_argument("--l_mlpnorm2", type=str, default="none")

    parser.add_argument("--num_layers", type=int, default=6)

    parser.add_argument("--nosv_uselinv", dest="sv_uselinv", action="store_false")
    parser.add_argument("--nosv_tailact", dest="sv_tailact", action="store_false")
    parser.add_argument("--nosv_res", dest="sv_res", action="store_false")
    parser.add_argument("--sv_numlayer", type=int, default=1)
    parser.add_argument("--sv_norm", type=str, default="none")


    parser.add_argument("--noel_uselinv", dest="el_uselinv", action="store_false")
    parser.add_argument("--el_uselins", action="store_true")
    parser.add_argument("--noel_tailact", dest="el_tailact", action="store_false")
    parser.add_argument("--el_numlayer", type=int, default=2)
    parser.add_argument("--el_norm", type=str, default="none")
    parser.add_argument("--el_uses", action="store_true")

    parser.add_argument("--noconv_uselinv", dest="conv_uselinv", action="store_false")
    parser.add_argument("--noconv_tailact", dest="conv_tailact", action="store_false")
    parser.add_argument("--conv_numlayer", type=int, default=1)
    parser.add_argument("--conv_norm", type=str, default="none")

    parser.add_argument("--predlin_numlayer", type=int, default=1)
    parser.add_argument("--predlin_norm",
                        choices=["bn", "ln", "in", "none"],
                        default="none")

    parser.add_argument("--lexp",
                        type=str,
                        choices=["gauss", "gg", "mlp", "sin", "id"],
                        default="mlp")
    parser.add_argument("--lexp_layer", type=int, default=2)
    parser.add_argument("--lexp_norm",
                        type=str,
                        choices=["bn", "ln", "in", "none"],
                        default="ln")
    
    parser.add_argument("--outln", action="store_true")
    parser.add_argument("--pool", type=str, default="mean")
    parser.add_argument("--Tm", type=float, default=1)

    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--load', type=str, default=None)

    parser.add_argument('--use_pos', action="store_true")
    parser.add_argument('--align_size', type=int, default=1)
    args = parser.parse_args()
    print(args)
    if args.featdim < 0:
        args.featdim = args.hiddim
    if args.caldim < 0:
        args.caldim = args.hiddim
    return args


def buildModel(args, num_tasks, device, dataset, needcompile: bool=True):
    xembdims = []
    tx = dataset.x
    if tx is None:
        xembdims = [None]
    elif tx.dtype == torch.long:
        assert tx.dim() == 2
        assert torch.all(tx != 0)
        xembdims = (torch.max(tx, dim=0)[0] + 1).tolist()
    assert torch.all(dataset.edge_attr != 0)
    from utils import act_dict
    kwargs = {
        "elres": args.elres,
        "usesvmix": args.usesvmix,
        "gsizenorm": args.gsizenorm,
        "vmean": args.vmean,
        "vnorm": args.vnorm,
        "elvmean": args.elvmean,
        "elvnorm": args.elvnorm,
        "snorm": args.snorm,
        "basic": {
            "dropout": args.dp,
            "activation": act_dict[args.act],
        },
        "inputencoder": {
            "dataset": args.dataset,
            "laplacian": args.laplacian,
            "sqrtlambda": args.sqrtlambda, 
            "xemb": {
                "orthoinit": args.emborthoinit,
                "bn": args.embbn,
                "ln": args.embln,
                "dropout": args.embdp,
            },
            "lambdaemb": {
                "numlayer": args.lexp_layer,
                "norm": args.lexp_norm,
            },
            "normA": args.normA,
            "degreeemb": args.degreeemb,
            "lexp": args.lexp,
            "xembdims": xembdims,
            "decompnoise": args.decompnoise,
            "use_pos": args.use_pos
        },
        "l_model": {
            "numlayers": args.l_layers,
            "aggr": args.l_aggr,
            "combine": args.l_combine,
            "res": args.l_res,
            "mlpargs1": {
                "numlayer": args.l_mlplayers1,
                "norm": args.l_mlpnorm1,
                "tailact": args.l_mlptailact1,
            },
            "mlpargs2": {
                "numlayer": args.l_mlplayers2,
                "norm": args.l_mlpnorm2,
                "tailact": args.l_mlptailact2,
            }
        },
        "svmix": {
            "uselinv": args.sv_uselinv,
            "numlayer": args.sv_numlayer,
            "norm": args.sv_norm,
            "tailact": args.sv_tailact,
            "res": args.sv_res
        },
        "elproj": {
            "uselinv": args.el_uselinv,
            "uselins": args.el_uselins,
            "numlayer": args.el_numlayer if args.caldim==args.hiddim else max(1, args.el_numlayer),
            "norm": args.el_norm,
            "tailact": args.el_tailact,
            "uses": args.el_uses,
        },
        "conv": {
            "uselinv": args.conv_uselinv,
            "numlayer": args.conv_numlayer,
            "norm": args.conv_norm,
            "tailact": args.conv_tailact,
        },
        "predlin": {
            "numlayer": args.predlin_numlayer,
            "norm": args.predlin_norm,
            "tailact": False,
        },
        "outln": args.outln
    }
    kwargs["predlin"].update(kwargs["basic"])
    kwargs["svmix"].update(kwargs["basic"])
    kwargs["l_model"]["mlpargs1"].update(kwargs["basic"])
    kwargs["l_model"]["mlpargs2"].update(kwargs["basic"])
    kwargs["inputencoder"]["basic"] = kwargs["basic"]
    kwargs["conv"].update(kwargs["basic"])
    kwargs["elproj"].update(kwargs["basic"])
    print("num_task", num_tasks)
    if args.nodetask:
        args.pool = "none"
    model = PiOModel(args.featdim, args.caldim, args.hiddim, num_tasks, args.num_layers, args.pool, **kwargs)
    model = model.to(device)
    if args.use_y_scale:
        ys = torch.concat([data.y for data in dataset])
        model.register_buffer("ymean", torch.mean(ys))
        model.register_buffer("ystd", torch.std(ys))
    else:
        model.register_buffer("ymean", torch.tensor(0))
        model.register_buffer("ystd", torch.tensor(1))
    print(model)
    print(f"numel {sum([p.numel() for p in model.parameters()])}")
    if needcompile:
        return torch.compile(model)
    else:
        return model


def main():
    # Training settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parserarg()
    ### automatic dataloading and splitting
    datasets, split, evaluator, task = loaddataset(args.dataset)
    args.nodetask = "node" in task
    task = task.removeprefix("node")
    print(split, task)
    outs = []
    set_seed(0)
    if split.startswith("fold"):
        trn_ratio, val_ratio, tst_ratio = int(split.split("-")[-3]), int(
            split.split("-")[-2]), int(split.split("-")[-1])
        num_fold = trn_ratio + val_ratio + tst_ratio
        trn_ratio /= num_fold
        val_ratio /= num_fold
        tst_ratio /= num_fold
        num_data = len(datasets[0])
        idx = torch.randperm(num_data)
        splitsize = num_data // num_fold
        idxs = [
            torch.cat((idx[splitsize * _:], idx[:splitsize * _]))
            for _ in range(num_fold)
        ]
        num_trn = int(trn_ratio * num_data)
        num_val = int(val_ratio * num_data)
    for rep in range(args.seedoffset, args.seedoffset + args.repeat):
        set_seed(rep)
        if "fixed" == split:
            trn_d, val_d, tst_d = datasets
        elif split.startswith("fold"):
            idx = idxs[rep]
            trn_idx, val_idx, tst_idx = idx[:num_trn], idx[
                num_trn:num_trn + num_val], idx[num_trn + num_val:]
            trn_d, val_d, tst_d = datasets[0][trn_idx], datasets[0][
                val_idx], datasets[0][tst_idx]
        else:
            datasets, split, evaluator, task = loaddataset(args.dataset)
            trn_d, val_d, tst_d = datasets
        print(len(trn_d), len(val_d), len(tst_d))
        train_loader = DataLoader(trn_d,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=args.num_workers)
        train_eval_loader = DataLoader(trn_d,
                                       batch_size=args.testbatch_size,
                                       shuffle=False,
                                       drop_last=False,
                                       num_workers=args.num_workers)
        valid_loader = DataLoader(val_d,
                                  batch_size=args.testbatch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=args.num_workers)
        test_loader = DataLoader(tst_d,
                                 batch_size=args.testbatch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=args.num_workers)
        print(f"split {len(trn_d)} {len(val_d)} {len(tst_d)}")
        model = buildModel(args, trn_d.num_tasks, device, trn_d, args.compile)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(args.beta, 0.999))
        if args.load is not None:
            print(f"mod/{args.load}.{rep}.pt")
            loadparams = torch.load(f"mod/{args.load}.{rep}.pt",
                                    map_location="cpu")
            print(model.load_state_dict(loadparams, strict=False))
            loadoptparams = torch.load(f"mod/{args.load}.opt.{rep}.pt",
                                    map_location="cpu")
            print(optimizer.load_state_dict(loadoptparams))
        scheduler0 = optim.lr_scheduler.LinearLR(optimizer, 1e-3, 1, args.warmstart*len(train_loader))
        scheduler1 = CosineAnnealingWarmRestarts(optimizer, args.cosstep*len(train_loader), eta_min=args.minlr, T_mult=args.Tm, K=args.K)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler0, scheduler1], milestones=[(args.conststep+args.warmstart)*len(train_loader)])
        valid_curve = []
        test_curve = []
        train_curve = []

        ampscaler = torch.cuda.amp.GradScaler() if args.amp else None
        lossfn = get_criterion(task, args).to(device)

        for epoch in range(1, args.epochs + 1):
            t1 = time.time()
            loss = train(lossfn, model, device, train_loader,
                         optimizer, task, ampscaler, args.advloss, scheduler, args.gradclipnorm, args)
            print(
                f"Epoch {epoch} train time : {time.time()-t1:.1f} loss: {loss:.2e}",
                flush=True)

            t1 = time.time()
            train_perf = 0.0 #eval(model, device, train_eval_loader, evaluator, args.amp, args)
            valid_perf = eval(model, device, valid_loader, evaluator, args.amp, args)
            if len(test_loader) > 0:
                test_perf = eval(model, device, test_loader, evaluator, args.amp, args)
            else:
                test_perf = 0.0
            print(
                f" test time : {time.time()-t1:.1f} Train {train_perf} Validation {valid_perf} Test {test_perf}",
                flush=True)
            train_curve.append(loss)
            valid_curve.append(valid_perf)
            test_curve.append(test_perf)
            if np.isnan(test_perf):
                break
            if args.save is not None:
                torch.save(model.state_dict(),
                                   f"mod/{args.save}.{rep}.pt")
                torch.save(optimizer.state_dict(),
                                   f"mod/{args.save}.opt.{rep}.pt")
            if epoch == 1:
                print(f"GPU memory {torch.cuda.max_memory_allocated()/1024/1024/1024:.2f}")
        if 'cls' in task:
            best_val_epoch = np.argmax(
                np.array(valid_curve) + np.arange(len(valid_curve)) * 1e-15)
            best_train = min(train_curve)
        else:
            best_val_epoch = np.argmin(
                np.array(valid_curve) - np.arange(len(valid_curve)) * 1e-15)
            best_train = min(train_curve)

        print(
            f'Best @{best_val_epoch} validation score: {valid_curve[best_val_epoch]:.4f} Test score: {test_curve[best_val_epoch]:.4f}', flush=True
        )
        outs.append([
            best_val_epoch, valid_curve[best_val_epoch],
            test_curve[best_val_epoch]
        ])
    print(outs)
    print(f"all runs: ", end=" ")
    for _ in np.average(outs, axis=0):
        print(_, end=" ")
    for _ in np.std(outs, axis=0):
        print(_, end=" ")
    print()


if __name__ == "__main__":
    main()
    print("", end="", flush=True)
    os._exit(os.EX_OK)