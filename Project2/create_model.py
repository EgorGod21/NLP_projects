import sys
sys.path.append('models')


def create_model(model_name, n_head, n_embd, block_size, dropout, vocab_size, n_layer, device):
    if model_name == "pre_ln":
        import pre_ln
        return pre_ln.GPTLanguageModel(n_head, n_embd, block_size, dropout, vocab_size, n_layer, device)
    elif model_name == "parallel":
        import parallel
        return parallel.GPTLanguageModel(n_head, n_embd, block_size, dropout, vocab_size, n_layer, device)
    elif model_name == "sas":
        import sas
        return sas.GPTLanguageModel(n_head, n_embd, block_size, dropout, vocab_size, n_layer, device)
    elif model_name == "sas_p":
        import sas_p
        return sas_p.GPTLanguageModel(n_head, n_embd, block_size, dropout, vocab_size, n_layer, device)
    else:
        raise ValueError(f"Model with name {model_name} does not exist ")