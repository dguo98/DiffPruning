# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import glob
import json
import logging
import os
import random
from IPython import embed

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sklearn

from transformers import (
	WEIGHTS_NAME,
	AdamW,
	AlbertConfig,
	AlbertForSequenceClassification,
	AlbertTokenizer,
	BertConfig,
	BertForSequenceClassification,
	BertTokenizer,
	DistilBertConfig,
	DistilBertForSequenceClassification,
	DistilBertTokenizer,
	FlaubertConfig,
	FlaubertForSequenceClassification,
	FlaubertTokenizer,
	RobertaConfig,
	RobertaForSequenceClassification,
	RobertaTokenizer,
	XLMConfig,
	XLMForSequenceClassification,
	XLMRobertaConfig,
	XLMRobertaForSequenceClassification,
	XLMRobertaTokenizer,
	XLMTokenizer,
	XLNetConfig,
	XLNetForSequenceClassification,
	XLNetTokenizer,
	get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors


try:
	from torch.utils.tensorboard import SummaryWriter
except ImportError:
	from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
	(
		tuple(conf.pretrained_config_archive_map.keys())
		for conf in (
			BertConfig,
			XLNetConfig,
			XLMConfig,
			RobertaConfig,
			DistilBertConfig,
			AlbertConfig,
			XLMRobertaConfig,
			FlaubertConfig,
		)
	),
	(),
)

MODEL_CLASSES = {
	"bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
	"xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
	"xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
	"roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
	"distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
	"albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
	"xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
	"flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)


def concrete_stretched(alpha, l=0., r = 1.):
	u = torch.zeros_like(alpha).uniform_().clamp_(0.0001, 0.9999)
	s = (torch.sigmoid(u.log() - (1-u).log() + alpha)).detach()
	u = s*(r-l) + l
	t = u.clamp(0, 1000)
	z = t.clamp(-1000, 1)
	dz_dt = (t < 1).float().to(alpha.device).detach()
	dt_du = (u > 0).float().to(alpha.device).detach()
	du_ds = r - l
	ds_dalpha = (s*(1-s)).detach()
	dz_dalpha = dz_dt*dt_du*du_ds*ds_dalpha
	return z.detach(), dz_dalpha.detach()

def get_valid_metric(result, task):
	if task == "cola":
		return result["mcc"]
	elif task == "sst-2":
		return result["acc"]
	elif task == "mrpc":
		return result["f1"]
	elif task == "sts-b":
		return result["spearmanr"]
	elif task == "qqp":
		return result["f1"]
	elif task == "mnli":
		return result["acc"]
	elif task == "mnli-mm":
		return result["acc"]
	elif task == "qnli":
		return result["acc"]
	elif task == "rte":
		return result["acc"]
	elif task == "hans":
		return result["acc"]
	else:
		raise KeyError(task)


def train(args, train_dataset, model, tokenizer):
	""" Train the model """
	if args.local_rank in [-1, 0]:
		tb_writer = SummaryWriter()

	best_val_metric = -10000
	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

	if args.max_steps > 0:
		t_total = args.max_steps
		args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
	else:
		t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ["bias", "LayerNorm.weight"]
	bert_params = {}
	finetune_params = []
	alpha_params = []
	print("args.local-rank=", args.local_rank)


	for n,p in model.named_parameters():
		p0 = torch.zeros_like(p.data).copy_(p) #original BERT
		p1 = torch.zeros_like(p.data) #params to be fine-tuned
		p1.requires_grad = True

		p1.grad = torch.zeros_like(p.data)
		alpha = torch.zeros_like(p.data) + args.alpha_init
		alpha.requires_grad = True
		alpha.grad = torch.zeros_like(p.data)
		if args.local_rank != -1 or args.n_gpu > 1:
			name = "module." + n
		else:
			name = n
		bert_params[name] = [p0, p1, alpha]
		finetune_params.append(bert_params[name][1])
		alpha_params.append(bert_params[name][2])
	model_device = list(model.named_parameters())[0][1].device
	debug_name = list(bert_params.keys())[0]

	if args.per_params_alpha == 1:
		per_params_alpha = {}
		for n, p in model.named_parameters():
			alpha = torch.zeros((1)).to(model_device) + args.alpha_init
			alpha.requires_grad=True
			alpha.grad = torch.zeros_like(alpha)
			if args.local_rank != -1 or args.n_gpu > 1:
				name = "module." + n
			else:
				name = n
			per_params_alpha[name] = alpha
			alpha_params.append(alpha)

	assert args.per_params_alpha == 0 or args.per_layer_alpha == 0, "Only support per params alpha OR per layer alpha"
	optimizer_grouped_parameters = [
		{
			"params": [p[1] for n, p in bert_params.items() if not any(nd in n for nd in no_decay) and p[1].requires_grad is True],
			"weight_decay": args.weight_decay,
		},
		{"params": [p[1] for n, p in bert_params.items() if any(nd in n for nd in no_decay) and p[1].requires_grad is True], "weight_decay": 0.0},
	]


	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	alpha_optimizer = AdamW(alpha_params, lr = 0.1, eps=args.adam_epsilon)

	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
	)

	scheduler2 = get_linear_schedule_with_warmup(
		alpha_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
	)

	# Check if saved optimizer or scheduler states exist
	if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
		os.path.join(args.model_name_or_path, "scheduler.pt")
	):
		# Load in optimizer and scheduler states
		optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
		scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

	if args.fp16:
		try:
			from apex import amp
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
		model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

	# multi-gpu training (should be after apex fp16 initialization)
	if args.n_gpu > 1:
		model = torch.nn.DataParallel(model)

	# Distributed training (should be after apex fp16 initialization)
	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(
			model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
		)

	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info(
		"  Total train batch size (w. parallel, distributed & accumulation) = %d",
		args.train_batch_size
		* args.gradient_accumulation_steps
		* (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
	)
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)

	global_step = 0
	epochs_trained = 0
	steps_trained_in_current_epoch = 0
	# Check if continuing training from a checkpoint
	if os.path.exists(args.model_name_or_path):
		# set global_step to global_step of last saved checkpoint from model path
		try:
			global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
		except ValueError:
			global_step = 0
		epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
		steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

		logger.info("  Continuing training from checkpoint, will skip to saved global_step")
		logger.info("  Continuing training from epoch %d", epochs_trained)
		logger.info("  Continuing training from global step %d", global_step)
		logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

	tr_loss, logging_loss = 0.0, 0.0
	model.zero_grad()
	train_iterator = trange(
		epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
	)
	set_seed(args)	# Added here for reproductibility
	print("model=",model)
	print("modelkeys=", model.state_dict().keys())
	total_layers = 14 if "base" in args.model_name_or_path else 26
	if args.sparsity_penalty_per_layer is None:
		sparsity_pen = [args.sparsity_pen] * total_layers  # NB(anon)
	else:
		sparsity_pen = args.sparsity_penalty_per_layer
		assert len(sparsity_pen) == total_layers,  "invalid sparsity penalty per layer: # of layers mismatch"

	modelname = args.model_type
	# get sparsity penalty
	def get_layer_ind(n):
		if modelname == "xlnet":
			if "transformer.word_embedding" in n:
				ind=0
			elif "transformer.layer"  in n:
				ind = int(n.replace("transformer.layer.", "").split(".")[0])+1
			else:
				ind = total_layers-1

		else:
			if "%s.embeddings"%modelname in n:
				ind = 0
			elif "%s.encoder.layer"%modelname in n:
				ind = int(n.replace("%s.encoder.layer."%modelname, "").split(".")[0]) + 1
			else:
				ind = total_layers - 1
		return ind

	def one_pass_concrete_stretched(x_alpha, ll,rr):
		z = []
		z_grad = []
		n_len = len(x_alpha)
		for i in range(4):
			l = (n_len//4) * i
			r = (n_len//4)*(i+1)
			if i == 3:
				r = max(r, n_len)
			z_, z_grad_ = concrete_stretched(x_alpha[l:r], args.concrete_lower, args.concrete_upper)
			assert z_.size(0) == r-l
			z.append(z_)
			z_grad.append(z_grad_)
		z = torch.cat(z, dim=0)
		z_grad = torch.cat(z_grad, dim=0)
		assert z.size(0) == x_alpha.size(0)
		return z, z_grad

	for _ in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
		epoch_steps = len(epoch_iterator)
		print("epoch_steps = ", epoch_steps)
		if args.save_steps > epoch_steps:
			args.save_steps = epoch_steps // 2
		for step, batch in enumerate(epoch_iterator):

			# Skip past any already trained steps if resuming training
			if steps_trained_in_current_epoch > 0:
				steps_trained_in_current_epoch -= 1
				continue
			nonzero_params = 0
			grad_params = {}
			if args.concrete_lower == 0:
				log_ratio = 0
			else:
				log_ratio = np.log(-args.concrete_lower / args.concrete_upper)

			# HACK(anon): sparsity_pen_for_14_layers
			# l0_pen = 0
			l0_pen = [0] * total_layers
			l0_pen_sum = 0


			if args.per_params_alpha:
				per_params_z = {}
				per_params_z_grad = {}
			#print("bert_params[", debug_name, "]=", bert_params[debug_name])
			torch.manual_seed(0)
			all_alpha = []
			all_pp_alpha = []
			all_add = []
			for n, p in model.named_parameters():
				if not "classifier" in n:
					all_alpha.append(bert_params[n][2].reshape(-1))
					all_add.append(bert_params[n][1].reshape(-1))
					all_pp_alpha.append(per_params_alpha[n].reshape(-1))
			all_alpha = torch.cat(all_alpha, dim=0)
			all_pp_alpha = torch.cat(all_pp_alpha, dim=0)
			all_add = torch.cat(all_add, dim=0)
			all_z, all_z_grad = one_pass_concrete_stretched(all_alpha, args.concrete_lower,
												   args.concrete_upper)
			all_z2_, all_z2_grad_ = one_pass_concrete_stretched(all_pp_alpha, args.concrete_lower, args.concrete_upper)
			
			l = 0
			nl = 0
			for n, p in model.named_parameters():
				if n not in bert_params:
					print(" n not in bert_params")
					embed()
				assert(n in bert_params)
				if "classifier" in n:
					nonzero_params += p.numel()
					p.data.copy_(bert_params[n][0].data + bert_params[n][1].data)
				else:
					r = l + p.numel()
					nr = nl + 1

					params_z = all_z2_[nl]
					params_z_grad = all_z2_grad_[nl]

					per_params_z[n] = params_z
					per_params_z_grad[n] = params_z_grad

					z = all_z[l:r].reshape_as(p)
					z_grad = all_z_grad[l:r].reshape_as(p)

					ind = get_layer_ind(n)
					b1 = bert_params[n][1]
					l0_pen[ind] += torch.sigmoid(bert_params[n][2] - log_ratio).sum()

					z2 =  per_params_z[n]

					grad_params[n] = [b1  * z2, z * z2, z_grad, b1 * z]

					l0_pen[ind] += torch.sigmoid(per_params_alpha[n] - log_ratio).sum()

					p.data.copy_(bert_params[n][0].data + (z2*z).data*b1.data)
					l = r
					nl = nr
			model.train()
			batch = tuple(t.to(args.device) for t in batch)
			inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
			if args.model_type != "distilbert":
				inputs["token_type_ids"] = (
					batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
				)  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
			outputs = model(**inputs)
			loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

			if args.n_gpu > 1:
				loss = loss.mean()	# mean() to average on multi-gpu parallel training
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps

			if args.fp16:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()

			tr_loss += loss.item()
			if (step + 1) % args.gradient_accumulation_steps == 0 or (
				# last step in epoch but step is always smaller than gradient_accumulation_steps
				len(epoch_iterator) <= args.gradient_accumulation_steps
				and (step + 1) == len(epoch_iterator)
			):

				if args.per_layer_alpha == 1:
					per_layer_alpha.grad.zero_()

				for n, p in model.named_parameters():
					if p.grad is None:
						continue
					if "classifier" in n:
						bert_params[n][1].grad.copy_(p.grad.data)
					else:
						bert_params[n][1].grad.copy_(p.grad.data * grad_params[n][1].data)
						bert_params[n][2].grad.copy_(p.grad.data * grad_params[n][0].data *
													 grad_params[n][2].data)

						per_params_alpha[n].grad.copy_(torch.sum(p.grad.data * grad_params[n][3].data *
									per_params_z_grad[n].data))

				sum_l0_pen = 0
				for i in range(total_layers):
					sum_l0_pen += (sparsity_pen[i] * l0_pen[i]).sum()
				sum_l0_pen.sum().backward()
				
				"""
				if args.fp16:
					torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
				else:
					torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
				"""
				torch.nn.utils.clip_grad_norm_(finetune_params, args.max_grad_norm)
				torch.nn.utils.clip_grad_norm_(alpha_params, args.max_grad_norm)
				optimizer.step()
				alpha_optimizer.step()
				scheduler.step()  # Update learning rate schedule
				#model.zero_grad()
				params_norm = [0, 0, 0, 0, 0, 0]
				# TODO(anon): think about expecation when we have multiplication of alpha
				exp_z = 0
				for n, p in bert_params.items():
					"""
					params_norm[0] += p[2].sum().item()
					params_norm[1] += p[2].norm().item()**2
					params_norm[2] += p[2].grad.norm().item()**2
					params_norm[3] += torch.sigmoid(p[2]).sum().item()
					params_norm[4] += p[2].numel()
					# params_norm[5] += (grad_params[n][1] > 0).float().sum().item()
					if args.per_params_alpha == 1:
						exp_z += (torch.sigmoid(p[2]).sum() * torch.sigmoid(per_params_alpha[n])).item()
					else:
						exp_z += torch.sigmoid(p[2]).sum().item()
					"""

					p[1].grad.zero_()
					p[2].grad.zero_()

				#mean_exp_z = exp_z / params_norm[4]


				if args.per_params_alpha == 1:
					for n,p in per_params_alpha.items():
						p.grad.zero_()
				"""
				if (step + 1)% 100 == 0:
					print("outdated average prob: %.4f, new average prob: %.4f, (!)empirical prob: %.4f, alpha_norm: %.4f, alpha_grad_norm: %.8f, alpha_avg: %.4f, l0_pen: %.2f, \n" %
						  (params_norm[3]/params_norm[4], mean_exp_z, nonzero_params/params_norm[4],
						   params_norm[1]**0.5, params_norm[2]**0.5, params_norm[0]/params_norm[4],
						   l0_pen_sum))
				"""
				global_step += 1

				if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
					logs = {}
					"""
					if (
						args.local_rank == -1 and args.evaluate_during_training
					):	# Only evaluate when single GPU otherwise metrics may not average well
						results = evaluate(args, model, tokenizer)
						for key, value in results.items():
							eval_key = "eval_{}".format(key)
							logs[eval_key] = value
					"""

					loss_scalar = (tr_loss - logging_loss) / args.logging_steps
					learning_rate_scalar = scheduler.get_lr()[0]
					logs["learning_rate"] = learning_rate_scalar
					logs["loss"] = loss_scalar
					logging_loss = tr_loss

					for key, value in logs.items():
						tb_writer.add_scalar(key, value, global_step)
					print(json.dumps({**logs, **{"step": global_step}}))

				if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
					# Save model checkpoint
					# HACK(anon)
					os.system("rm -rf %s/checkpoint-*" % (args.output_dir))
					model_to_save = (
						model.module if hasattr(model, "module") else model
					)  # Take care of distributed/parallel training

					info_dict = {"model": model.state_dict(), "bert_params": bert_params}
					torch.save(info_dict, os.path.join(args.output_dir, "checkpoint-last-info.pt"))
					logger.info("Saving all training information: bert params, z, nonzero_params at time step  %d to checkpoint-last-info.pt" % global_step)

					logs = {}
					results = evaluate(args, model, tokenizer)
					for key, value in results.items():
						eval_key = "eval_{}".format(key)
						logs[eval_key] = value
					print(json.dumps({**logs, **{"step": global_step}}))

					val_metric = get_valid_metric(results, args.task_name)
					if val_metric > best_val_metric:
						best_val_metric = val_metric
						torch.save(info_dict, os.path.join(args.output_dir, "checkpoint-best-info.pt"))
						logger.info("Saving all training information: bert params, z, nonzero_params to checkpoint-best-info.pt")

					"""
					if global_step % (args.save_steps * 5) == 0:
						# HACK(anon): prevent memory exceed ?!
						model_to_save.save_pretrained(output_dir)
						tokenizer.save_pretrained(output_dir)

						torch.save(args, os.path.join(output_dir, "training_args.bin"))
						logger.info("Saving model checkpoint to %s", output_dir)

						torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
						torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
						logger.info("Saving optimizer and scheduler states to %s", output_dir)
					"""

			if args.max_steps > 0 and global_step > args.max_steps:
				epoch_iterator.close()
				break
		if args.max_steps > 0 and global_step > args.max_steps:
			train_iterator.close()
			break

	if args.local_rank in [-1, 0]:
		tb_writer.close()

	return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
	# Loop to handle MNLI double evaluation (matched, mis-matched)
	eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
	eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

	results = {}
	for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
		eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

		if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
			os.makedirs(eval_output_dir)

		args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
		# Note that DistributedSampler samples randomly
		eval_sampler = SequentialSampler(eval_dataset)
		eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

		# multi-gpu eval
		if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
			model = torch.nn.DataParallel(model)

		# Eval!
		logger.info("***** Running evaluation {} *****".format(prefix))
		logger.info("  Num examples = %d", len(eval_dataset))
		logger.info("  Batch size = %d", args.eval_batch_size)
		eval_loss = 0.0
		nb_eval_steps = 0
		preds = None
		out_label_ids = None
		for batch in tqdm(eval_dataloader, desc="Evaluating"):
			model.eval()
			batch = tuple(t.to(args.device) for t in batch)

			with torch.no_grad():
				inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
				if args.model_type != "distilbert":
					inputs["token_type_ids"] = (
						batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
					)  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
				outputs = model(**inputs)
				tmp_eval_loss, logits = outputs[:2]

				eval_loss += tmp_eval_loss.mean().item()
			nb_eval_steps += 1
			if preds is None:
				preds = logits.detach().cpu().numpy()
				out_label_ids = inputs["labels"].detach().cpu().numpy()
			else:
				preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
				out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

		eval_loss = eval_loss / nb_eval_steps
		if args.output_mode == "classification":
			preds = np.argmax(preds, axis=1)
		elif args.output_mode == "regression":
			preds = np.squeeze(preds)
		result = compute_metrics(eval_task, preds, out_label_ids)
		results.update(result)
		if args.task_name == "mnli":
			results.update({"%s_acc" % eval_task: result["acc"]})
			result = results  # HACK


		output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
		with open(output_eval_file, "w") as writer:
			logger.info("***** Eval results {} *****".format(prefix))
			for key in sorted(result.keys()):
				logger.info("  %s = %s", key, str(result[key]))
				writer.write("%s = %s\n" % (key, str(result[key])))
	return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
	if args.local_rank not in [-1, 0] and not evaluate:
		torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

	processor = processors[task]()
	output_mode = output_modes[task]
	# Load data features from cache or dataset file
	cached_features_file = os.path.join(
		args.data_dir,
		"cached_{}_{}_{}_{}".format(
			"dev" if evaluate else "train",
			list(filter(None, args.model_name_or_path.split("/"))).pop(),
			str(args.max_seq_length),
			str(task),
		),
	)
	if os.path.exists(cached_features_file) and not args.overwrite_cache:
		logger.info("Loading features from cached file %s", cached_features_file)
		features = torch.load(cached_features_file)
	else:
		logger.info("Creating features from dataset file at %s", args.data_dir)
		label_list = processor.get_labels()
		if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
			# HACK(label indices are swapped in RoBERTa pretrained model)
			label_list[1], label_list[2] = label_list[2], label_list[1]
		examples = (
			processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
		)
		features = convert_examples_to_features(
			examples,
			tokenizer,
			label_list=label_list,
			max_length=args.max_seq_length,
			output_mode=output_mode,
			pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
			pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
			pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
		)
		if args.local_rank in [-1, 0]:
			logger.info("Saving features into cached file %s", cached_features_file)
			torch.save(features, cached_features_file)

	if args.local_rank == 0 and not evaluate:
		torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

	# Convert to Tensors and build dataset
	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
	all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
	if output_mode == "classification":
		all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
	elif output_mode == "regression":
		all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

	dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
	return dataset


def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument(
		"--data_dir",
		default=None,
		type=str,
		required=True,
		help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
	)
	parser.add_argument(
		"--model_type",
		default=None,
		type=str,
		required=True,
		help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
	)
	parser.add_argument(
		"--model_name_or_path",
		default=None,
		type=str,
		required=True,
		help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
	)
	parser.add_argument(
		"--task_name",
		default=None,
		type=str,
		required=True,
		help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
	)
	parser.add_argument(
		"--output_dir",
		default=None,
		type=str,
		required=True,
		help="The output directory where the model predictions and checkpoints will be written.",
	)

	# Other parameters
	parser.add_argument(
		"--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
	)
	parser.add_argument(
		"--tokenizer_name",
		default="",
		type=str,
		help="Pretrained tokenizer name or path if not the same as model_name",
	)
	parser.add_argument(
		"--cache_dir",
		default="",
		type=str,
		help="Where do you want to store the pre-trained models downloaded from s3",
	)
	parser.add_argument(
		"--max_seq_length",
		default=128,
		type=int,
		help="The maximum total input sequence length after tokenization. Sequences longer "
		"than this will be truncated, sequences shorter will be padded.",
	)

	parser.add_argument(
		"--sparsity_pen",
		default=0.000000125,
		type=float,
		help="Sparsity penalty.",
	)

	parser.add_argument(
		"--sparsity_penalty_per_layer",
		default=None,
		type=float,
		nargs="+",
		help="Sparsity penalty per layer.",
	)


	parser.add_argument(
		"--temp",
		default=1.,
		type=float,
		help="Temperature.",
	)

	parser.add_argument(
		"--concrete_lower",
		default=-1.5,
		type=float,
		help="Temperature.",
	)

	parser.add_argument(
		"--concrete_upper",
		default=1.5,
		type=float,
		help="Temperature.",
	)
	parser.add_argument(
		"--fix_layer",
		default=-1,
		type=int,
		help="whether to fix layers"
	)
	parser.add_argument(
		"--alpha_init",
		default=5,
		type=int,
		help="Alpha init value",
	)


	parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
	parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
	parser.add_argument(
		"--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
	)
	parser.add_argument(
		"--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
	)

	parser.add_argument(
		"--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
	)
	parser.add_argument(
		"--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
	)
	parser.add_argument(
		"--gradient_accumulation_steps",
		type=int,
		default=1,
		help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
	parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
	parser.add_argument(
		"--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
	)
	parser.add_argument(
		"--max_steps",
		default=-1,
		type=int,
		help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
	)
	parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

	parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
	parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
	parser.add_argument(
		"--eval_all_checkpoints",
		action="store_true",
		help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
	)
	parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
	parser.add_argument(
		"--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
	)
	parser.add_argument(
		"--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
	)
	parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

	parser.add_argument(
		"--fp16",
		action="store_true",
		help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
	)
	parser.add_argument(
		"--fp16_opt_level",
		type=str,
		default="O1",
		help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
		"See details at https://nvidia.github.io/apex/amp.html",
	)
	parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
	parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
	parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
	parser.add_argument("--per_layer_alpha", type=int, default=0, help="Per layer alpha")
	parser.add_argument("--per_params_alpha", type=int, default=1, help="Per params alpha")
	args = parser.parse_args()
	print("parse args!!=", args)
	print(args.sparsity_penalty_per_layer)

	if (
		os.path.exists(args.output_dir)
		and os.listdir(args.output_dir)
		and args.do_train
		and not args.overwrite_output_dir
	):
		raise ValueError(
			"Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
				args.output_dir
			)
		)

	# Setup distant debugging if needed
	if args.server_ip and args.server_port:
		# Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
		import ptvsd

		print("Waiting for debugger attach")
		ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
		ptvsd.wait_for_attach()

	# Setup CUDA, GPU & distributed training
	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
	else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		torch.distributed.init_process_group(backend="nccl")
		args.n_gpu = 1
	args.device = device

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
	)
	logger.warning(
		"Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
		args.local_rank,
		device,
		args.n_gpu,
		bool(args.local_rank != -1),
		args.fp16,
	)

	# Set seed
	set_seed(args)

	# Prepare GLUE task
	args.task_name = args.task_name.lower()
	if args.task_name not in processors:
		raise ValueError("Task not found: %s" % (args.task_name))
	processor = processors[args.task_name]()
	args.output_mode = output_modes[args.task_name]
	label_list = processor.get_labels()
	num_labels = len(label_list)

	# Load pretrained model and tokenizer
	if args.local_rank not in [-1, 0]:
		torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

	args.model_type = args.model_type.lower()
	config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
	config = config_class.from_pretrained(
		args.config_name if args.config_name else args.model_name_or_path,
		num_labels=num_labels,
		finetuning_task=args.task_name,
		cache_dir=args.cache_dir if args.cache_dir else None,
	)
	tokenizer = tokenizer_class.from_pretrained(
		args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
		do_lower_case=args.do_lower_case,
		cache_dir=args.cache_dir if args.cache_dir else None,
	)
	model = model_class.from_pretrained(
		args.model_name_or_path,
		from_tf=bool(".ckpt" in args.model_name_or_path),
		config=config,
		cache_dir=args.cache_dir if args.cache_dir else None,
	)

	if args.local_rank == 0:
		torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

	model.to(args.device)

	logger.info("Training/evaluation parameters %s", args)

	# Training
	if args.do_train:
		train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
		global_step, tr_loss = train(args, train_dataset, model, tokenizer)
		logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

	# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
	if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
		# Create output directory if needed
		if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
			os.makedirs(args.output_dir)

		logger.info("Saving model checkpoint to %s", args.output_dir)
		# Save a trained model, configuration and tokenizer using `save_pretrained()`.
		# They can then be reloaded using `from_pretrained()`
		model_to_save = (
			model.module if hasattr(model, "module") else model
		)  # Take care of distributed/parallel training
		model_to_save.save_pretrained(args.output_dir)
		tokenizer.save_pretrained(args.output_dir)

		# Good practice: save your training arguments together with the trained model
		torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

		# Load a trained model and vocabulary that you have fine-tuned
		model = model_class.from_pretrained(args.output_dir)
		tokenizer = tokenizer_class.from_pretrained(args.output_dir)
		model.to(args.device)

	# Evaluation
	results = {}
	if args.do_eval and args.local_rank in [-1, 0]:
		tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
		checkpoints = [args.output_dir]
		if args.eval_all_checkpoints:
			checkpoints = list(
				os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
			)
			logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
		logger.info("Evaluate the following checkpoints: %s", checkpoints)
		for checkpoint in checkpoints:
			global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
			prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

			model = model_class.from_pretrained(checkpoint)
			model.to(args.device)
			result = evaluate(args, model, tokenizer, prefix=prefix)
			result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
			results.update(result)

	return results


if __name__ == "__main__":
	main()

