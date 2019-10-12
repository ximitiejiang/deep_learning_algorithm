#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:30:02 2019

@author: ubuntu
"""
from model.runner_lib import Runner

def test_ssdvgg16_antsbees():
    runner = Runner(cfg_path = './cfg_classifier_ssdvgg16_antsbees.py')
    runner.train()
    runner.evaluate()
    
    
if __name__ == "__main__":
    test_ssdvgg16_antsbees()