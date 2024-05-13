import os
import random

token = "<token>"
answer = "<answer>"

def getFile():
    path = '../linux'
    file_list = [""]
    
    # Add all C files into a list
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith('.c'):
                file_list.append(os.path.join(root, name))

    #choose a file at random and return it
    chosen_one = random.choice(file_list)
    return chosen_one


def main():
    path = getFile()
    print(path)


    maxLines = 100 #Maximum number of lines from a single file
    with open(path, "r") as reader:
        output_file = "data_batch1.txt"
        with open(output_file, "w") as writer:
            lines = 0
            for line in reader:
                line = line.strip()
                if line.startswith('//'):
                    continue
                elif:
                    #Add support for multiline comment
                else:
                    if(lines <= maxLines):
                        lines = lines + 1
                        # Replace words with token, not sure how to do that
                        toBeReplaced = " should be chosen randomly"



                        tokenized_line = line.replace(toBeReplaced, token)
                        tokenized_line = tokenized_line + answer + toBeReplaced
                        # Write to output file
                        writer.write(tokenized_line)
                    else:
                        break



    ##Läsa in rader, ignorera kommentarer
    ##Replacea något ord med token - logik för det

main()