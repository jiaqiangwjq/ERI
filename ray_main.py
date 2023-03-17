import torch
import argparse
from model import *
from dataset import Multimodal_Datasets
from torch.utils.data import DataLoader
from functools import partial
from torch.optim import Adam
from util import calu_PCC
from ray_config import *
from ray import air, tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def collate_fn(data):
    audio_token_list = []
    audio_mask_list = []
    visual_list = []
    visual_mask_list = []
    groundtruth_list = []

    for unit in data:
        audio_token_list.append(unit[0])
        audio_mask_list.append(unit[1])
        visual_list.append(unit[2])
        visual_mask_list.append(unit[3])
        groundtruth_list.append(unit[4])

    audio_token = torch.cat(audio_token_list, dim=0)
    audio_mask = torch.cat(audio_mask_list, dim=0)
    visual = torch.cat(visual_list, dim=0)
    visual_mask = torch.cat(visual_mask_list, dim=0)
    groundtruth = torch.cat(groundtruth_list, dim=0)
    
    return audio_token, audio_mask, visual, visual_mask, groundtruth


def Train(config):
    opts.d_ff = config["d_ff"]
    opts.lr = config["lr"]
    opts.batch_size = config["batch_size"]
    
    train_dataset = Multimodal_Datasets(opts, split_type="train")
    val_dataset = Multimodal_Datasets(opts, split_type="val")
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    model = TE(opts, opts.d_model * opts.modal_num)

    if opts.use_cuda:
        model = model.cuda()
    optimizer = Adam(model.parameters(), opts.lr)
    criterion = torch.nn.MSELoss()
    model.train()

    for _ in range(10):
        results = []
        truths = []
        for batch_data in train_loader :
            audio, audio_mask, visual, visual_mask, truth = batch_data
            model.zero_grad()
            if opts.use_cuda:
                audio, audio_mask, visual, visual_mask, truth = audio.cuda(), audio_mask.cuda(), visual.cuda(), visual_mask.cuda(), truth.cuda()
            pred = model(visual, visual_mask)
            results.append(pred.cpu().detach()) 
            truths.append(truth.cpu())
            loss = criterion(pred, truth)
            loss.backward()
            optimizer.step()

        results = torch.cat(results, dim=0)
        truths = torch.cat(truths, dim=0)
        pcc_train = calu_PCC(results.numpy(), truths.numpy())[1]
        
        model.eval()
        loader = val_loader
        results = []
        truths = []

        with torch.no_grad():
            for batch_data in loader:
                audio, audio_mask, visual, visual_mask, truth = batch_data
                if opts.use_cuda:
                    audio, audio_mask, visual, visual_mask, truth = audio.cuda(), audio_mask.cuda(), visual.cuda(), visual_mask.cuda(), truth.cuda()
                pred = model(visual, visual_mask)
                results.append(pred.cpu().detach()) 
                truths.append(truth.cpu().detach())

        results = torch.cat(results, dim=0)
        truths = torch.cat(truths, dim=0)
        pcc_val = calu_PCC(results.numpy(), truths.numpy())[1]
        state_dict = (model.state_dict(), optimizer.state_dict())
        checkpoint = Checkpoint.from_dict(
            dict(weights=state_dict)
        )
        session.report(
            metrics={"pcc_train": pcc_train, "pcc_val": pcc_val},
            checkpoint=checkpoint,
        )


def main(num_samples, max_num_epochs):
    config = {
        "lr": tune.loguniform(1e-5, 2e-4),
        "d_ff": tune.choice([2**x for x in range(8, 11)]),
        "batch_size": tune.choice([2**x for x in range(3, 6)]),
        "should_checkpoint": True,
    }
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="pcc_val",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
        brackets=1,)
    hyperopt_search = HyperOptSearch(config, metric="pcc_val", mode="max")
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(Train),
            resources={"cpu":4, "gpu":1},
        ),
        run_config = air.RunConfig(
            local_dir="/data02/lixin/ray_results_enet+fabnet+dan_all_data_pcc_loss",
            name="log",
            checkpoint_config = air.CheckpointConfig(
                checkpoint_score_attribute="pcc_val",
                num_to_keep=2,
            ),
        ),
        tune_config = tune.TuneConfig(
            num_samples = num_samples,
            search_alg = hyperopt_search,
            scheduler = scheduler,
        ),
    )
    result = tuner.fit()


if __name__ == '__main__':
    main(num_samples=100, max_num_epochs=15)
    

