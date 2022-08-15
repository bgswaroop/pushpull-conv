import torch


def compute_map_score(train_hash_codes, train_ground_truths, query_hash_codes, query_ground_truths,
                      top_k = None, device=None):
    """
    credits: https://github.com/weixu000/DSH-pytorch/blob/906399c3b92cf8222bca838c2b2e0e784e0408fa/utils.py#L58
    :param train_hash_codes: An input tensor of dims (num_train, hash_length) with binary values
    :param train_ground_truths: An input tensor of dims (num_train,)
    :param query_hash_codes: An input tensor of dims (num_query, hash_length) with binary values
    :param query_ground_truths: An input tensor of dims (num_query,)
    :param top_k: Number of samples to consider for evaluation of the mAP score
    :param device: torch.device()
    :return: map_score
    """

    train_hash_codes = train_hash_codes.to(device)
    train_ground_truths = train_ground_truths.to(device)
    query_hash_codes = query_hash_codes.to(device)
    query_ground_truths = query_ground_truths.to(device)

    AP = []  # average precision
    num_samples = torch.arange(1, train_hash_codes.size(0) + 1).to(device)
    for i in range(query_hash_codes.size(0)):
        query_label, query_hash = query_ground_truths[i], query_hash_codes[i]
        hamming_dist_between_query_and_training_data = torch.sum((query_hash != train_hash_codes).long(), dim=1)
        ranking = hamming_dist_between_query_and_training_data.argsort()
        correct = (query_label == train_ground_truths[ranking]).float().to(device)

        if top_k is None:
            P = torch.cumsum(correct, dim=0) / num_samples  # precision vector
        else:
            correct = correct[:top_k]
            P = torch.cumsum(correct, dim=0) / top_k

        if torch.sum(correct) == 0:
            AP.append(torch.Tensor([0]).to(device))
        else:
            AP.append(torch.sum(P * correct) / torch.sum(correct))

    map_score = torch.mean(torch.Tensor(AP))
    return map_score
