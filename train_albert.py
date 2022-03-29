import os
from statistics import mode
from pytest import param
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch

import math
from torch import cdist, nn
import torch.nn.functional as F 
import argparse
import numpy as np
import logging
import sys
from os.path import join
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from csr_mhqa.argument_parser import default_train_parser, complete_default_train_parser, json_to_argv
from csr_mhqa.data_processing import Example, InputFeatures, DataHelper
from csr_mhqa.utils import *

from models.HGN import *
from transformers import get_linear_schedule_with_warmup
from torch.nn import MSELoss


from csr_mhqa.utils import result_to_file

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

#########################################################################
# Initialize arguments
##########################################################################
parser = default_train_parser()

logger.info("IN CMD MODE")
args_config_provided = parser.parse_args(sys.argv[1:])
if args_config_provided.config_file is not None:
    argv = json_to_argv(args_config_provided.config_file) + sys.argv[1:]
else:
    argv = sys.argv[1:]
args = parser.parse_args(argv)
args = complete_default_train_parser(args)   #完整的参数

logger.info('-' * 100)
logger.info('Input Argument Information')
logger.info('-' * 100)
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

#########################################################################
# Read Data
#########################################################################
helper = DataHelper(gz=True, config=args)

# Set train datasets
train_dataloader = helper.train_loader
dev_example_dict = helper.dev_example_dict
dev_feature_dict = helper.dev_feature_dict
dev_dataloader = helper.dev_loader

#########################################################################
# Initialize Model
##########################################################################

cached_config_file = join(args.exp_name, 'cached_config.bin')
if os.path.exists(cached_config_file):
    cached_config = torch.load(cached_config_file)
    encoder_path = join(args.exp_name, cached_config['encoder'])
    model_path = join(args.exp_name, cached_config['model'])
    learning_rate = cached_config['lr']
    start_epoch = cached_config['epoch']
    best_joint_f1 = cached_config['best_joint_f1']
    logger.info("Loading encoder from: {}".format(encoder_path))
    logger.info("Loading model from: {}".format(model_path))
else:
    encoder_path = None
    model_path = None
    start_epoch = 0
    best_joint_f1 = 0
    learning_rate = args.learning_rate


teacher_encoder_file = join(args.exp_name, 'teacher_model/encoder.pkl')
teacher_model_file = join(args.exp_name, 'teacher_model/model.pkl')

if os.path.exists(teacher_encoder_file) and os.path.exists(teacher_model_file):
    logger.info("exist teacher file!")

# Set teacher Encoder and Model
encoder, config = load_encoder_model(args.encoder_name_or_path, args.model_type)


#设置encoder_name_or_path 为albert-base-v2
encoder_s, config_s= load_encoder_model_s(args.encoder_name_or_path_student, args.model_type)

model = HierarchicalGraphNetwork(config=args) 
model_s = HierarchicalGraphNetwork(config=args) 

if teacher_encoder_file is not None:
    print("load teacher encoder")
    encoder.load_state_dict(torch.load(teacher_encoder_file))

if teacher_model_file is not None:
    print("load teacher model")
    model.load_state_dict(torch.load(teacher_model_file))


if encoder_path is not None:  #设置加载预训练权重
    print("load encoder_path")
    encoder_s.load_state_dict(torch.load(encoder_path))


if model_path is not None:  #这里的权重设置为空
    print("load model_path")
    model_s.load_state_dict(torch.load(model_path))

encoder.to(args.device)
model.to(args.device)

encoder_s.to(args.device)
model_s.to(args.device)

#设置encoder、model模型不更新
#设置encoder_s,model_s模型更新

for _, value in encoder.named_parameters():
    value.requires_grad=False
for _, value in model.named_parameters():
    value.requires_grad=False



t_model={}

t_encoder_total_params =  sum(p.numel() for p in  encoder.parameters())

t_encoder_trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

t_model_total_params =  sum(p.numel() for p in model.parameters())

t_model_trainable_params = sum(p.numel() for p in model.parameters()  if p.requires_grad)

t_model['encoder_total_params']=t_encoder_total_params
t_model['encoder_trainable_params']=t_encoder_trainable_params
t_model['model_total_params']=t_model_total_params
t_model['model_trainable_params']=t_model_trainable_params


logger.info("writing teacher model")
result_to_file(t_model,"/home/user12/HGN/t_model.json")

#print model



s_model={}
student_encoder_total_params =  sum(p.numel() for p in  encoder_s.parameters())
student_encoder_trainable_params = sum(p.numel() for p in encoder_s.parameters() if p.requires_grad)
student_model_total_params =  sum(p.numel() for p in model_s.parameters())
student_model_trainable_params = sum(p.numel() for p in model_s.parameters()  if p.requires_grad)

s_model['encoder_total_params']=student_encoder_total_params
s_model['encoder_trainable_params']=student_encoder_trainable_params
s_model['model_total_params']=student_model_total_params
s_model['model_trainable_params']=student_model_trainable_params

logger.info("writing student model")
result_to_file(s_model,"/home/user12/HGN/s_model.json")




_, _, tokenizer_class = MODEL_CLASSES[args.model_type]
tokenizer = tokenizer_class.from_pretrained(args.encoder_name_or_path,
                                            do_lower_case=args.do_lower_case)

#########################################################################
# Evalaute if resumed from other checkpoint
##########################################################################

if encoder_path is not None and model_path is not None:
    output_pred_file = os.path.join(args.exp_name, 'pred.epoch.json')
    output_eval_file = os.path.join(args.exp_name, 'eval.epoch.txt')
    prev_metrics, prev_threshold = eval_model(args, encoder_s, model_s,
                                              dev_dataloader, dev_example_dict, dev_feature_dict,
                                              output_pred_file, output_eval_file, args.dev_gold_file)
    logger.info("Best threshold for prev checkpoint: {}".format(prev_threshold))
    for key, val in prev_metrics.items():
        logger.info("{} = {}".format(key, val))

#########################################################################
# Get Optimizer
##########################################################################
if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


# optimizer = get_optimizer(encoder, model, args, learning_rate, remove_pooler=False)
loss_mse = MSELoss()
optimizer = get_optimizer(encoder_s, model_s, args, learning_rate, remove_pooler=False)

#调用amp.initialize()不能调用分布式函数
# if args.fp16:
#     try:
#         from apex import amp
#     except ImportError:
#         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
#     models, optimizer = amp.initialize([encoder, model], optimizer, opt_level=args.fp16_opt_level)
#     assert len(models) == 2
#     encoder, model = models

if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    models, optimizer = amp.initialize([encoder_s, model_s], optimizer, opt_level=args.fp16_opt_level)
    assert len(models) == 2
    encoder_s, model_s = models

# Distributed training (should be after apex fp16 initialization)
if args.local_rank != -1: #把当前model推送到local_rank指定的gpu的内存上
    encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[args.local_rank],
                                                        output_device=args.local_rank,
                                                        find_unused_parameters=True)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank,
                                                      find_unused_parameters=True)

    encoder_s = torch.nn.parallel.DistributedDataParallel(encoder_s, device_ids=[args.local_rank],
                                                        output_device=args.local_rank,
                                                        find_unused_parameters=True)

    model_s = torch.nn.parallel.DistributedDataParallel(model_s, device_ids=[args.local_rank],
                                                      output_device=args.local_rank,
                                                      find_unused_parameters=True)

    

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=args.warmup_steps,
                                            num_training_steps=t_total)


#########################################################################
# launch training
##########################################################################
global_step = 0

# loss_name = ["loss_total", "loss_span", "loss_type", "loss_sup", "loss_ent", "loss_para"]

loss_name =["tol_loss","d_loss","loss_ditill","loss_hard","loss_graph_state"]
tr_loss, logging_loss = [0] * len(loss_name), [0]* len(loss_name)


if args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter(args.exp_name)

encoder_s.zero_grad()
model_s.zero_grad()

def distillation_loss(y,  teacher_scores, T):
    d_loss = nn.KLDivLoss(reduction='mean')(F.log_softmax(y / T, dim=1),
                                                      F.softmax(teacher_scores / T, dim=1)) * T * T
    return  d_loss


train_iterator = trange(start_epoch, start_epoch+int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
for epoch in train_iterator:

    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    train_dataloader.refresh()
    dev_dataloader.refresh()

    for step, batch in enumerate(epoch_iterator):
        encoder_s.train()
        model_s.train()
        
        att_loss=0
        rep_loss=0

        loss_list=[]
    
        inputs = {'input_ids':      batch['context_idxs'],  # batch_size,len(input_ids)
                  'attention_mask': batch['context_mask'],  # batch_size,len(input_ids)
                  'token_type_ids': batch['segment_idxs'] if args.model_type in ['bert', 'xlnet'] else None}  # XLM don't use segment_ids

        batch['context_encoding'] = encoder(**inputs)[0] #shape (bs,input_ids,hidden_dim)  抽取bert最后一层隐藏层向量
        batch['context_mask'] = batch['context_mask'].float().to(args.device)        # context_mask shape (bs,input_ids) context_mask 标记哪些是有效tokens,哪些是PAD
        

        # Cal KG distill
        outputs_teacher=encoder(**inputs)
        teacher_reps, teacher_atts = outputs_teacher[2] , outputs_teacher[3]

        outputs_student = encoder_s(**inputs)
        student_reps, student_atts = outputs_student[2] , outputs_student[3]

        teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]  # speedup 1.5x
        teacher_atts = [teacher_att.detach() for teacher_att in teacher_atts]

        teacher_layer_num = len(teacher_atts)
        student_layer_num = len(student_atts)

        assert teacher_layer_num % student_layer_num == 0

        layers_per_block = int(teacher_layer_num / student_layer_num)

        new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                    for i in range(student_layer_num)]


        for student_att, teacher_att in zip(student_atts, new_teacher_atts):
            student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(args.device),
                                        student_att)
            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(args.device),
                                        teacher_att)
            att_loss += loss_mse(student_att, teacher_att)

        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
        new_student_reps = student_reps

        for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
            rep_loss += loss_mse(student_rep, teacher_rep)

        loss_ditill  =   rep_loss + att_loss


        #start=(batch_size,input_len)  
        #end=(batch_size,input_len) 
        # q_type=(batch_size,q_type_num)
        # paras=(batch_size,max_num_para,2)   
        # sents=(batch_size,max_num_sent,2), 
        # ents=(batch_size,max_num_entity)


        start, end, q_type, paras, sents, ents, _, _,graph_state = model(batch, return_yp=True)  # 教师模型输出的结果

        start_s, end_s, q_type_s, paras_s, sents_s, ents_s, _, _,graph_state_s = model_s(batch, return_yp=True)  #学生模型输出的结果

        #d_loss is prediction_loss between teacher and student

        # y is student logits  teacher_scores is teacher logits
        d_loss  = [] 

        d_loss.append(distillation_loss(start_s,start,args.T)) #set T
        d_loss.append(distillation_loss(end_s,end,args.T))
        d_loss.append(distillation_loss(q_type_s,q_type,args.T))
        d_loss.append(distillation_loss(paras_s,paras,args.T))
        d_loss.append(distillation_loss(sents_s,sents,args.T))
        d_loss.append(distillation_loss(ents_s,ents,args.T))

        #d_loss.append(distillation_loss(graph_state_s,graph_state,args.T))  #两个图向量的分布

        d_loss.append(loss_mse(graph_state_s, graph_state))

        graph_state_loss= d_loss[6]  #0.023

        d_loss=sum(d_loss[:6])

        loss_hard_list = compute_loss(args, batch, start_s, end_s, paras_s, sents_s, ents_s, q_type_s) # 学生模型输出和hard_label的损失
        
        loss_hard= loss_hard_list[0] #这个太大了，不加入

        #tol_loss =  args.d_lambda*d_loss + args.ditill_lambda*loss_ditill + args.hard_lambda*loss_hard + args.graph_lambda*graph_state_loss  
        #d_loss  = 6   loss_ditill =0.1531  loss_hard= 22  graph_state_loss=0.023

        #tol_loss = 2*(d_loss/d_loss.detach()) + 5*(loss_ditill/loss_ditill.detach()) + loss_hard/loss_hard.detach()+5*(graph_state_loss/graph_state_loss.detach())
    
        tol_loss = d_loss/d_loss.detach()+ loss_ditill/loss_ditill.detach() + graph_state_loss/graph_state_loss.detach()

        loss_list.append(tol_loss)
        loss_list.append(d_loss)
        loss_list.append(loss_ditill)
        loss_list.append(loss_hard)
        loss_list.append(graph_state_loss)
        
        del batch 
        if args.n_gpu > 1:
            for loss in loss_list:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            for loss in loss_list:
                loss = loss / args.gradient_accumulation_steps
        if args.fp16: #run here
            with amp.scale_loss(loss_list[0], optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss_list[0].backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)


        for idx in range(len(loss_name)):
            if not isinstance(loss_list[idx], int):
                tr_loss[idx] += loss_list[idx].data.item()
            else:
                tr_loss[idx] += loss_list[idx]
        
        if (step + 1) % args.gradient_accumulation_steps == 0:  #only update student model
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            encoder_s.zero_grad()
            model_s.zero_grad()
            global_step += 1

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                avg_loss = [ (_tr_loss - _logging_loss) / (args.logging_steps*args.gradient_accumulation_steps)
                             for (_tr_loss, _logging_loss) in zip(tr_loss, logging_loss)]

                loss_str = "step[{0:6}] " + " ".join(['%s[{%d:.5f}]' % (loss_name[i], i+1) for i in range(len(avg_loss))])
                logger.info(loss_str.format(global_step, *avg_loss))
                logging_loss = tr_loss.copy()

        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break
    #每次蒸馏结束后，在student 上进行测试，把表现好的student模型保留
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        output_pred_file = os.path.join(args.exp_name, f'pred.epoch_{epoch+1}.json')  #每epoch预测一次
        output_eval_file = os.path.join(args.exp_name, f'eval.epoch_{epoch+1}.txt')
        metrics, threshold = eval_model(args, encoder_s, model_s,
                                        dev_dataloader, dev_example_dict, dev_feature_dict,
                                        output_pred_file, output_eval_file, args.dev_gold_file)

        if metrics['joint_f1'] >= best_joint_f1:
            best_joint_f1 = metrics['joint_f1']
            torch.save({'epoch': epoch+1,
                        'lr': scheduler.get_lr()[0],
                        'encoder': 'encoder.pkl',
                        'model': 'model.pkl',
                        'best_joint_f1': best_joint_f1,
                        'threshold': threshold},
                       join(args.exp_name, f'cached_config.bin')
            )
        torch.save({k: v.cpu() for k, v in encoder_s.state_dict().items()},
                    join(args.exp_name, f'encoder_{epoch+1}.pkl'))
        torch.save({k: v.cpu() for k, v in model_s.state_dict().items()},
                    join(args.exp_name, f'model_{epoch+1}.pkl'))

        for key, val in metrics.items():
            tb_writer.add_scalar(key, val, epoch)

if args.local_rank in [-1, 0]:
    tb_writer.close()


#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 9999  train.py --config_file configs/train.albert.json
