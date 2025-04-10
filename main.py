from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import math
import random
import numpy as np
import torch.nn.functional as F
from preprocess.DataLoader_FF import DataLoader_multi_source_and_target
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.vit_model import *
from models.loss import *
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
logger = logging.getLogger(__name__)
from sklearn.metrics import roc_auc_score
from models.ap import compute_average_precision
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000

def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp) / len(pred_idx)

def save_model(args, model, is_adv=False):
    model_to_save = model.module if hasattr(model,'module') else model
    if not is_adv:
        model_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint.bin" % args.name)
    else:
        model_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint_adv.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", os.path.join(args.output_dir, args.dataset))

def setup(args):

    model = vit_base_patch16_224_in21k()
    model.load_state_dict(torch.load(args.pretrained_dir))
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, args.num_classes)
    num_params = count_parameters(model)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model

def valid(args, model,log_file, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    all_preds = []
    all_probs=[]
    all_label=[]
    train_acc = 0
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = CrossEntropyLoss()
    for step, batch in tqdm(enumerate(epoch_iterator)):
        x = batch['img'].to(args.device, non_blocking=True).float()
        y = batch['label'].to(args.device, non_blocking=True).long()
        with torch.no_grad():
            logits, _ = model(x)
            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            acc = compute_accuray(F.log_softmax(logits, dim=1), y)
            train_acc += acc
        all_preds += preds.detach().cpu().numpy().tolist()
        all_probs += probs.detach().cpu().numpy().tolist()
        all_label += y.detach().cpu().numpy().tolist()
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    auc = roc_auc_score(all_label, all_probs)
    ap=compute_average_precision(all_label, all_probs)
    accuracy = train_acc / len(test_loader)

    logger.info("\n")
    logger.info("Validation Results of: %s" % args.name)
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid AUC: %2.5f" % auc)
    logger.info("Valid AP: %2.5f" % ap)
    logger.info("Valid Accuracy: %2.5f" % accuracy)
    #写入到日志中
    write_to_log(f"test/AUC:{auc}", log_file)
    write_to_log(f"test/AP:{ap}", log_file)
    write_to_log(f"test/Accuracy:{accuracy}", log_file)
    return auc,accuracy,ap

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def write_to_log(message, file):
    print(message)
    file.write(message + '\n')

def train(args, model):
    if args.local_rank in [-1, 0]:
        log_dir = os.path.join("logs", args.dataset, args.name)
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "training_log.txt")
        log_file = open(log_file_path, 'w')
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    target_dataset = DataLoader_multi_source_and_target(phase='train', domain='target', image_size=args.img_size,
                                                        n_frames=args.n_frames,
                                                        forgery=args.target_list)
    test_dataset = DataLoader_multi_source_and_target(phase='test', domain='source', image_size=args.img_size,
                                                      n_frames=args.n_frames,
                                                      forgery=args.source_list)
    source_dataset = DataLoader_multi_source_and_target(phase='train', domain='source', image_size=args.img_size,
                                                        n_frames=args.n_frames,
                                                        forgery=args.source_list)

    source_loader = torch.utils.data.DataLoader(source_dataset,
                                                batch_size=args.train_batch_size,
                                                shuffle=True,
                                                collate_fn=source_dataset.collate_fn,
                                                num_workers=4,
                                                pin_memory=True,
                                                drop_last=True,
                                                worker_init_fn=source_dataset.worker_init_fn
                                                )
    target_loader = torch.utils.data.DataLoader(target_dataset,
                                                batch_size=args.train_batch_size,
                                                shuffle=True,
                                                collate_fn=target_dataset.collate_fn,
                                                num_workers=4,
                                                pin_memory=True,
                                                drop_last=True,
                                                worker_init_fn=target_dataset.worker_init_fn
                                                )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.eval_batch_size,
                                              shuffle=True,
                                              collate_fn=test_dataset.collate_fn,
                                              num_workers=4,
                                              pin_memory=True,
                                              worker_init_fn=test_dataset.worker_init_fn
                                              )
    ad_net = AdversarialNetwork(model.head.in_features, model.head.in_features // 4)
    ad_net.to(args.device)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer_ad= torch.optim.Adam(ad_net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_ad = WarmupCosineSchedule(optimizer_ad, warmup_steps=args.warmup_steps, t_total=t_total)

    elif args.decay_type == "linear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_ad = WarmupLinearSchedule(optimizer_ad, warmup_steps=args.warmup_steps, t_total=t_total)

    model.zero_grad()
    ad_net.zero_grad()
    set_seed(args)
    losses = AverageMeter()
    best_ap = 0
    best_auc = 0.5
    best_acc = 0
    len_source = len(source_loader)
    len_target = len(target_loader)

    for global_step in tqdm(range(1, t_total)):
        model.train()
        if (global_step - 1) % (len_source - 1) == 0:
            iter_source = iter(source_loader)
        if (global_step - 1) % (len_target - 1) == 0:
            iter_target = iter(target_loader)

        data_source = next(iter_source)
        data_target = next(iter_target)
        x_s= data_source['img'].to(args.device, non_blocking=True).float()
        y_s = data_source['label'].to(args.device, non_blocking=True).long()
        x_t = data_target['img'].to(args.device, non_blocking=True).float()
        y_t = data_target['label'].to(args.device, non_blocking=True).long()
        logits_s,features_s=model(x_s)
        logits_t, features_t=model(x_t)
        logits=torch.cat((logits_s, logits_t), 0)
        y=torch.cat((y_s, y_t), 0)

        loss_fct = CrossEntropyLoss()
        loss_clc = loss_fct(logits.view(-1, args.num_classes), y.view(-1))

        loss_ad_global = adv(torch.cat((features_s[:, 0], features_t[:, 0]), 0), ad_net)

        loss = loss_clc + args.beta * loss_ad_global

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(ad_net.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        optimizer_ad.step()
        optimizer_ad.zero_grad()
        scheduler_ad.step()

        if global_step % args.eval_every == 0 and args.local_rank in [-1, 0] and global_step >= 500:
            # 记录
            write_to_log(f'train/Global Step: {global_step}------------------val begin------------------', log_file)
            write_to_log(f"train/Loss: {loss.item()}", log_file)
            write_to_log(f"train/loss_clc: {loss_clc.item()}", log_file)
            write_to_log(f"train/loss_ad_global: {loss_ad_global.item()}", log_file)
            write_to_log(f"train/lr: {scheduler.get_last_lr()[0]}", log_file)

            logger.info(" Train Batch size = %d", len(y_s))
            auc,acc,ap= valid(args, model, log_file, test_loader, global_step)
            if best_auc < auc:
                save_model(args, model)
                best_auc = auc

                if acc != 0.5:
                    best_acc = acc
                best_ap=ap
            model.train()
            logger.info("Current Best auc: %s" % best_auc)
            logger.info("Current Best ap: %s" % best_ap)
            logger.info("Current Best acc: %s" % best_acc)
    if args.local_rank in [-1, 0]:
        log_file.close()
    logger.info("Best auc: \t%s" % best_auc)
    logger.info("Best ap: \t%f" % best_ap)
    logger.info("Best acc: \t%s" % best_acc)
    logger.info("End Training!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", help="Which downstream task.")
    parser.add_argument("--source_list", help="Path of the training data.",type=str,default=['Deepfakes', 'Face2Face', 'FaceSwap','NeuralTextures'])
    parser.add_argument("--target_list", help="Path of the test data.",type=str,default=['Deepfakes', 'Face2Face', 'FaceSwap','NeuralTextures'])
    parser.add_argument("--num_classes", default=2, type=int,
                        help="Number of classes in the dataset.")
    parser.add_argument("--n_frames", default=8, type=int,
                        help="Number of frames in the dataset.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="preprocess/jx_vit_base_patch16_224_in21k-e5005f0a.pth",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=96, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--beta", default=1.0, type=float,
                        help="The importance of the adversarial loss.")
    parser.add_argument("--msa_layer", default=12, type=int,
                        help="The layer that incorporates local alignment.")
    parser.add_argument("--is_test", default=False, action="store_true",
                        help="If in test mode.")

    parser.add_argument("--learning_rate", default=1e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=5000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear","halfcosine","halflinear"], default="linear",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    set_seed(args)
    if args.is_test:
        valid(args)
    else:
        args, model = setup(args)
        model.to(args.device)
        train(args, model)


if __name__ == "__main__":
    main()


