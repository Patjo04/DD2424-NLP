import argparse
import torch
import lex
from network import Network

def temp_scaling(logits, temp):
    return logits / temp

def generate(model, ctx, ctx_len, out_len, temp=1.0):
    output = ctx
    while len(output) < out_len:
        k = min(ctx_len, len(output))
        ctx = (output[-k:] if k > 0 else []) + ['$']
        logits = model([ctx])
        logits = temp_scaling(logits, temp)
        word = model.logits_to_word(logits)
        output.append(word)
    return output

def main(ctx, ctx_len, model_path, out_len):
    model = Network.load(model_path)
    device = "cuda" if torch.cuda.is_available()\
            else "mps" if torch.backends.mps.is_available()\
            else "cpu"
    model._device = device
    ctx = ctx.split()
    tokens = generate(model, ctx, ctx_len, out_len)
    for token in tokens:
        # Reverse key lookup.
        key = reverse_lookup(token, [lex.Lexer.SHORT1, lex.Lexer.SHORT2, lex.Lexer.SHORT3])
        if key != None:
            token = key
        elif token[:2] == 'KW':
            token = token[3:].lower()
        elif token in lex.Lexer.VOCAB and token != 'IDENT':
            token = token.lower()
        end = '\n' if token == ';' else ' '
        print(token, end=end)

def reverse_lookup(token, dicts):
    for dict in dicts:
        for key, val in dict.items():
            if token == val:
                return key
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='generate.py')
    parser.add_argument('context')
    parser.add_argument('-m', '--model')
    parser.add_argument('-k', '--ctxlen', type=int, default=3)
    parser.add_argument('-n', '--outlen', type=int, default=100)
    args = parser.parse_args()
    model_path = args.model
    ctx = args.context
    ctx_len = args.ctxlen
    out_len = args.outlen
    main(ctx, ctx_len, model_path, out_len)