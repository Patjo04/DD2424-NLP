import random
#Startprogram: Skriv ut ditt namn och få ett hej tillbaka!
def main(namn):
    print("Hej hopp " + str(namn) + "!")
    print("Ditt lyckotal för nu är: " + str(random.randint(0, 2147483647)))

main(input())