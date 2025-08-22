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