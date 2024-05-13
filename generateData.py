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

    reader = open(path, "r")


    ##Läsa in rader, ignorera kommentarer
    ##Replacea något ord med token - logik för det

main()