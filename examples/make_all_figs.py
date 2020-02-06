#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
import os
import multiprocessing
import tqdm



def make_fig(s):
	os.system(f"python3 {os.path.dirname(os.path.abspath(__file__))}/make_{s}_fig.py")


if __name__ == '__main__':

	figures = ["xi","drude","fi","ratio","dre","material","fth"]
	pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
	for _ in tqdm.tqdm(pool.imap_unordered(make_fig, figures), total = len(figures)):
		pass