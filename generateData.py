#Program to generate data files from code in the linux github repo.
#For the program to work the linux repo has to be in the same directory as this repo

import os
import random
from lex import Lexer


def getFile():
    path = '.'
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

def main():
    samples = []
    lexer = Lexer()
    used_files = set() #Keep track of read files so that each file is used only once
    num_samples = 50 ##1000000
    dropout = 0.5
    num_samples_pre_dropout = num_samples / dropout
    
    while len(samples) < num_samples_pre_dropout: #Read lines from files until batch is filled

        input_file = getFile()
        if input_file in used_files:
            continue
        used_files.add(input_file)

        with open(input_file, "r") as reader:
            tokens = lexer.lex_file(reader)
            tokens = list(map(lambda tup: tup[1], tokens))
            for feature, label in generate_samples(tokens, 5, 2):
                sample = f'{feature}, {label}\n'
                samples.append(sample)
    
    random.shuffle(samples)
    samples = samples[:num_samples]
    
    training_proportion = 0.8
    num_training = int(training_proportion * len(samples))
    training = samples[:num_training]
    test = samples[num_training:]

    train_file = f"data/train.txt"
    test_file = f"data/test.txt"
    with open(train_file, "w") as writer:
        for sample in training:
            writer.writelines([sample])
    with open(test_file, "w") as writer:
        for sample in test:
            writer.writelines([sample])

if __name__ == '__main__':
    main()