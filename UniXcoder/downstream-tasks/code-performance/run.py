# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys 
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from model import Model
from loss import PerformanceMetricLoss
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.optim import AdamW
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
							  RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)

class InputFeatures(object):
	"""A single training/test features for a example."""
	def __init__(self,
				 code_tokens,
				 code_ids,
				 code_perf,
	):
		self.code_tokens = code_tokens
		self.code_ids = code_ids
		self.code_perf = code_perf
		
def convert_examples_to_features(js, tokenizer, args):
	"""convert examples to token ids"""
	code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
	code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
	code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
	code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
	padding_length = args.code_length - len(code_ids)
	code_ids += [tokenizer.pad_token_id]*padding_length
	code_perf = js['code_perf']
	return InputFeatures(code_tokens, code_ids, code_perf)

class TextDataset(Dataset):
	def __init__(self, tokenizer, args, file_path=None):
		self.examples = []
		data = []
		with open(file_path) as f:
			if "jsonl" in file_path:
				for line in f:
					line = line.strip()
					js = json.loads(line)
					data.append(js)
		
		for js in data:
			self.examples.append(convert_examples_to_features(js, tokenizer, args))
		
		if 'train' in file_path:
			for idx, example in enumerate(self.examples[:3]):
				logger.info("** Example **")
				logger.info("idx: {}".format(idx))
				logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
				logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
				logger.info("code_perf: {}".format(example.code_perf))
	
	def __len__(self):
		return len(self.examples)
	
	def __getitem__(self, i):
		return (torch.tensor(self.examples[i].code_ids), self.examples[i].code_perf)

def set_seed(seed=42):
	random.seed(seed)
	os.environ['PYHTONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True

def train(args, model: Model, tokenizer):
	""" Train the model """
	#get training dataset
	train_dataset = TextDataset(tokenizer, args, args.train_data_file)
	train_sampler = RandomSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

	#get optimizer and scheduler
	optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)
	
	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
	logger.info("  Total train batch size  = %d", args.train_batch_size)
	logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)

	model.zero_grad()

	model.train()
	tr_num, tr_loss = 0, 0 
	for idx in range(args.num_train_epochs):
		for step, batch in enumerate(train_dataloader):
			code_inputs = batch[0].to(args.device)
			code_perfs = batch[1].to(args.device)

			code_vec = model(code_inputs=code_inputs)

			loss_fct = PerformanceMetricLoss()
			loss = loss_fct(code_vec, code_perfs)

			tr_loss += loss.item()
			tr_num += 1

			if (step + 1) % 10 == 0:
				logger.info("epoch {} step {} loss {}".format(idx, step + 1, round(tr_loss / tr_num, 5)))
				tr_num = 0
				tr_loss = 0
			
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
			optimizer.step()
			optimizer.zero_grad()
			scheduler.step() 


def main():
	parser = argparse.ArgumentParser()
	
	## Required parameters
	parser.add_argument("--train_data_file", default=None, type=str, help="The input training data file(a .jsonl file)")
	parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
	parser.add_argument("--test_data_file", default=None, type=str, help="An optional input test data file to test the model(a jsonl file).")
	parser.add_argument("--model_name_or_path", default=None, type=str, help="The model checkpoint for weights initialization.")
	parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
	parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
	parser.add_argument("--do_test", action='store_true', help="Whether to run eval on the test set.")  
	parser.add_argument("--train_batch_size", default=4, type=int, help="Batch size for training.")
	parser.add_argument("--eval_batch_size", default=4, type=int, help="Batch size for evaluation.")
	parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
	parser.add_argument("--num_train_epochs", default=1, type=int, help="Total number of training epochs to perform.")
	parser.add_argument("--code_length", default=256, type=int, help="Optional Code input sequence length after tokenization.") 

	parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

	args = parser.parse_args()

	logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.n_gpu = torch.cuda.device_count()
	args.device = device
	logger.info("device: %s, n_gpu: %s",device, args.n_gpu)

	# Set seed
	set_seed(args.seed)

	# Build model
	tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
	model = RobertaModel.from_pretrained(args.model_name_or_path) 
	model = Model(model)
	model.to(args.device)

	logger.info("Training/evaluation parameters %s", args)

	if args.do_train:
		train(args, model, tokenizer)
	
	if args.do_eval:
		pass



if __name__ == "__main__":
	main()