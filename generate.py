import argparse
from network import Network

def temp_scaling(prob, temp):
    return prob # TODO

def generate(model, ctx, ctx_len, out_len):
    output = ctx
    while len(output) < out_len:
        k = min(ctx_len, len(output))
        ctx = (output[-k:] if k > 0 else []) + ['$']
        prob = model([ctx])
        prob = temp_scaling(prob, 1.0)
        word = model.prob_to_word(prob)
        output.append(word)
    return output

def main(ctx, ctx_len, model_path, out_len):
    model = Network.load(model_path)
    ctx = ctx.split()
    tokens = generate(model, ctx, ctx_len, out_len)
    for token in tokens:
        print(token, end=' ')

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