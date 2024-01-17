import torch


def compute_map_score(train_hash_codes, train_ground_truths, query_hash_codes, query_ground_truths, device=None,
                      return_as_float=False):
    """
    credits: https://github.com/weixu000/DSH-pytorch/blob/906399c3b92cf8222bca838c2b2e0e784e0408fa/utils.py#L58
    :param train_hash_codes: An input tensor of dims (num_train, hash_length) with binary values
    :param train_ground_truths: An input tensor of dims (num_train,)
    :param query_hash_codes: An input tensor of dims (num_query, hash_length) with binary values
    :param query_ground_truths: An input tensor of dims (num_query,)
    :param device: torch.device()
    :param return_as_float: (FALSE) return the scores as Tensor by default.
    :return: map_score considering top-k retrieved samples
    """

    train_hash_codes = train_hash_codes.to(device)
    train_ground_truths = train_ground_truths.to(device)
    query_hash_codes = query_hash_codes.to(device)
    query_ground_truths = query_ground_truths.to(device)

    num_samples = torch.arange(1, train_hash_codes.size(0) + 1).to(device)
    top_k_values = [train_hash_codes.size(0), 50, 100, 200, 500, 1000]
    AP = {f'top{k}': [] for k in top_k_values}  # average precision

    for i in range(query_hash_codes.size(0)):
        query_label, query_hash = query_ground_truths[i], query_hash_codes[i]
        hamming_dist_between_query_and_training_data = torch.sum((query_hash != train_hash_codes).long(), dim=1)
        ranking = hamming_dist_between_query_and_training_data.argsort()
        is_relevant_sample = (query_label == train_ground_truths[ranking]).float().to(device)

        for top_k in top_k_values:
            precision_at_k = torch.cumsum(is_relevant_sample[:top_k], dim=0) / num_samples[:top_k]
            precision_at_k_for_relevant_samples = precision_at_k[torch.where(is_relevant_sample[:top_k] == 1)]
            avg_precision = torch.nan_to_num(torch.mean(precision_at_k_for_relevant_samples), nan=0)
            AP[f'top{top_k}'].append(avg_precision)

    map_score = {}
    for key, value in AP.items():
        map_score[key] = torch.mean(torch.Tensor(AP[key]))

    if return_as_float:
        for key, value in map_score.items():
            map_score[key] = float(value)

    return map_score


def accuracy(y_hat: torch.Tensor, y: torch.Tensor, top_k: int = 1):
    """
    Compute accuracy for multi-class classification
    :param y_hat: Logits, Softmax scores, or the output of the final classification layer of the network
    :param y: True labels (expressed as indices, starting from 0)
    :param top_k: Top k labels to consider for classification.
    :return: top_k accuracy score
    """
    ranked_predictions = torch.argsort(y_hat, dim=1, descending=True)
    top_k_predictions = ranked_predictions[:, :top_k]
    top_k_results = torch.max(top_k_predictions == y.reshape(-1, 1), dim=1)[0]
    top_k_accuracy = torch.mean(top_k_results.to(float))
    return top_k_accuracy
