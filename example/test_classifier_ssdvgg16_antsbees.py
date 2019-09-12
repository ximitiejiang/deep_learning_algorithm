#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:30:02 2019

@author: ubuntu
"""

from model.runner_lib import Runner



def main():
    runner = Runner(cfg_path = '../model/cfg_classifier_ssdvgg16_antsbees.py')
    runner.train()
    runner.evaluate()
    
    
if __name__ == "__main__":
    main()