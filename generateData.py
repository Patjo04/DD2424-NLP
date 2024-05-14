#Program to generate data files from code in the linux github repo.
#For the program to work the linux repo has to be in the same directory as this repo

import os
import random

token = "<token>"
answer = "<answer>"
max_lines_per_file = 1000 #Maximum number of lines from a single file
lines_per_batch = 10000 # total number of lines in each data file

def getFile():
    path = '../linux'

    if not os.path.exists(path):
        raise FileNotFoundError(f"The linux repo must be in the same directory as this repo")

    file_list = [""]
    
    # Add all C files into a list
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith('.c'):
                file_list.append(os.path.join(root, name))

    #choose a file at random and return it
    chosen_one = random.choice(file_list)
    return chosen_one


def modifyLine(line):
    words = line.split()
    rep_idx = random.randint(0,len(words) -1)
    rep_word = words[rep_idx]
    words[rep_idx] = token
    return ' '.join(words) + f' {answer} {rep_word} \n'

def read_data(reader, writer, lines_in_batch):
    lines = 0
    in_multiline = False
    for line in reader:
        #Remove leading & trailing whitespaces.
        line = line.strip()

        #Skip all rows that includes part of a multiline comment
        if '/*' in line:
            in_multiline = True
            continue
        elif in_multiline:
            if '*/' in line:
                in_multiline = False
            continue

        #Remove single-line comments
        elif line.startswith('//'):
            continue
        elif '//' in line:
            line = line.split('//')[0]

        # skip lines with only one word
        if len(line.split()) < 2:
            continue

        # Modify and write line to output file
        if lines < max_lines_per_file and lines_in_batch < lines_per_batch :
            lines += 1
            lines_in_batch += 1
            writer.write(modifyLine(line))
        else:
            break
    return lines_in_batch


def main():
    used_files = set() #Keep track of read files so that each file is used only once
    for batch_num in range(1, 12):
        lines_in_batch = 0

        output_file = f"data/batch{batch_num}.txt"
        if batch_num == 11:
            output_file = "data/test.txt"

        with open(output_file, "w") as writer:            
            while lines_in_batch < lines_per_batch: #Read lines from files until batch is filled

                input_file = getFile()
                if input_file in used_files:
                    continue
                used_files.add(input_file)

                with open(input_file, "r") as reader:
                    lines_in_batch = read_data(reader, writer, lines_in_batch)
        print(f"batch {batch_num} done")
main()