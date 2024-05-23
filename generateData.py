#Program to generate data files from code in the linux github repo.
#For the program to work the linux repo has to be in the same directory as this repo

import os
import random
import argparse
from lex import Lexer


def getFile():
    path = '../linux'
    if not os.path.exists(path):
        raise FileNotFoundError(f"The linux repo must be in the same directory as this repo")

    file_list = []
    
    # Add all C files into a list
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith('.c'):
                file_list.append(os.path.join(root, name))

    #choose a file at random and return it
    chosen_one = random.choice(file_list)
    return chosen_one

def generate_samples(tokens, left_ctx_len, right_ctx_len):
    n = len(tokens)
    for i in range(left_ctx_len, n-right_ctx_len-1):
        label = tokens[i]
        left_start = i - left_ctx_len
        right_end = i + right_ctx_len + 1
        left_ctx = ' '.join(tokens[left_start:i])
        right_ctx = ' '.join(tokens[i+1:right_end])
        feature = '$'
        if left_ctx:
            feature = f'{left_ctx} {feature}'
        if right_ctx:
            feature = f'{feature} {right_ctx}'
        yield feature, label

def generate_samples_rand(tokens, min_ctx_len, max_ctx_len):
    n = len(tokens)
    i = 0
    while i < n:
        ctx_len = min_ctx_len + int((max_ctx_len - min_ctx_len + 1) * random.random())
        left_ctx_len = int((ctx_len + 1) * random.random())
        right_ctx_len = ctx_len - left_ctx_len
        if i < left_ctx_len:
            i += left_ctx_len
        
        tokens_left = n - i - 1
        if tokens_left < right_ctx_len:
            break

        label = tokens[i]
        left_start = i - left_ctx_len
        right_end = i + right_ctx_len + 1
        left_ctx = ' '.join(tokens[left_start:i])
        right_ctx = ' '.join(tokens[i+1:right_end])
        feature = '$'
        if left_ctx:
            feature = f'{left_ctx} {feature}'
        if right_ctx:
            feature = f'{feature} {right_ctx}'
        i += ctx_len
        yield feature, label

def main():
    parser = argparse.ArgumentParser(prog='generateData')
    parser.add_argument('left', type=int)
    parser.add_argument('right', type=int)
    parser.add_argument('-n', '--samples', type=int, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-r', '--random', action='store_true')
    args = parser.parse_args()
    left_ctx_len = args.left
    right_ctx_len = args.right
    num_samples = args.samples
    output_dir = args.output
    random_ctx_len = args.random

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    samples = []
    lexer = Lexer()
    used_files = set() #Keep track of read files so that each file is used only once
    dropout = 0.5
    num_samples_pre_dropout = num_samples / dropout
    encoding = 'utf-8'
    
    while len(samples) < num_samples_pre_dropout: #Read lines from files until batch is filled

        input_file = getFile()
        if input_file in used_files:
            continue
        used_files.add(input_file)

        with open(input_file, "r", encoding=encoding) as reader:
            tokens = lexer.lex_file(reader)
            tokens = list(map(lambda tup: tup[1], tokens))
            if random_ctx_len:
                samples2 = generate_samples_rand(tokens, left_ctx_len, right_ctx_len)
            else:
                samples2 = generate_samples(tokens, left_ctx_len, right_ctx_len)
            for feature, label in samples2:
                sample = f'{feature}, {label}\n'
                samples.append(sample)
    
    random.shuffle(samples)
    samples = samples[:num_samples]
    
    training_proportion = 0.8
    num_training = int(training_proportion * len(samples))
    training = samples[:num_training]
    test = samples[num_training:]

    train_file = os.path.join(output_dir, 'train.txt')
    test_file = os.path.join(output_dir, 'test.txt')
    with open(train_file, "w", encoding=encoding) as writer:
        for sample in training:
            writer.writelines([sample])
    with open(test_file, "w", encoding=encoding) as writer:
        for sample in test:
            writer.writelines([sample])

if __name__ == '__main__':
    main()
