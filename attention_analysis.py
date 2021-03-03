from lit_classes import GeceProdigyData, GectorBertModel, \
    GECE_ERROR_TYPES  # reuse the LIT classes as they're already packaged for analysis

from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
from lit_nlp.api import model as lit_model, dataset as lit_dataset
from tqdm import tqdm as normal_tqdm
from tqdm.notebook import tqdm as notebook_tqdm
from typing import List, Dict


def _get_results_from_model(model: lit_model.Model, data: lit_dataset.Dataset, notebook: bool) -> List[Dict]:
    tqdm = notebook_tqdm if notebook else normal_tqdm

    batch_size = model.max_minibatch_size()
    results = []
    for i in tqdm(range(0, len(data), batch_size), desc='processing batches'):
        batch_examples = data.examples[i:i + batch_size]
        results.extend(model.predict_minibatch(batch_examples))

    return results


def _perform_analysis(results: List[Dict], data: lit_dataset.Dataset, attention_heads: int, attention_layers: int,
                       max_len:int, prepended_token_count: int, notebook: bool):

    tqdm = notebook_tqdm if notebook else normal_tqdm
    total_head_count = attention_heads + 1  # one extra head representing the average
    head_avg_index = attention_heads

    layers = ['layer{}'.format(x) for x in range(0, attention_layers)] + ['layer_average']
    error_token_labels = {et: [] for et in GECE_ERROR_TYPES}
    error_token_values = {et: {layer: [[] for x in range(0, total_head_count)] for layer in layers}
                          for et in GECE_ERROR_TYPES}
    error_labels_argmax_guess = {et: {layer: [[] for x in range(0, total_head_count)] for layer in layers}
                                 for et in GECE_ERROR_TYPES}

    for result_dict, input_dict in tqdm(zip(results, data.examples), desc='processing results', total=len(results)):
        input_tokens = input_dict['input_tokens']
        processed_input_tokens = result_dict['input_tokens']
        assert (input_tokens == processed_input_tokens[1:])
        if layers[0] not in result_dict or len(input_tokens) + 1 >= result_dict[layers[0]].shape[1]:
            continue  # input was under min length (ignored) or over max length (truncated), don't bother with it
        for marking in input_dict['markings']:
            error_type = marking['error_type']
            labels_relative_to_error = [1 if idx in marking['cause_indices'] else 0 for idx in range(len(input_tokens))]

            # ideally the check to make sure  the error marking was inside the sentence tokens wouldn't be necessary...
            for error_token_index in (x+prepended_token_count for x in marking['error_indices'] if x < len(input_tokens)):
                error_token_labels[error_type].extend(labels_relative_to_error)
                for layer in layers:
                    for head_index in range(0, total_head_count - 1):
                        attention_for_tokens = result_dict[layer][head_index, error_token_index,
                                            range(prepended_token_count, (len(input_tokens) + prepended_token_count))]
                        max_attention_index = np.argmax(attention_for_tokens)
                        argmax_label_guess = [1 if idx == max_attention_index else 0 for idx
                                              in range(len(attention_for_tokens))]
                        error_token_values[error_type][layer][head_index].extend(attention_for_tokens)
                        error_labels_argmax_guess[error_type][layer][head_index].extend(argmax_label_guess)


                    attention_for_tokens = np.average(result_dict[layer], axis=0)[error_token_index,
                                            range(prepended_token_count, (len(input_tokens) + prepended_token_count))]
                    max_attention_index = np.argmax(attention_for_tokens)
                    argmax_label_guess = [1 if idx == max_attention_index else 0 for idx
                                          in range(len(attention_for_tokens))]
                    error_token_values[error_type][layer][head_avg_index].extend(attention_for_tokens)
                    error_labels_argmax_guess[error_type][layer][head_avg_index].extend(argmax_label_guess)



    pearson_results = {
        error_type: {
            layer: [] for layer in layers
        } for error_type in GECE_ERROR_TYPES
    }

    regression_results = {
        error_type: {
            layer: [] for layer in layers
        } for error_type in GECE_ERROR_TYPES
    }

    argmax_results = {
        error_type: {
            layer: [] for layer in layers
        } for error_type in GECE_ERROR_TYPES
    }

    for error_type in GECE_ERROR_TYPES:
        error_labels = error_token_labels[error_type]
        if len(error_labels) < 10:
            continue  # not enough data to compute anything for this type
        for layer in layers:
            for head_index in range(0, total_head_count):
                error_values = error_token_values[error_type][layer][head_index]
                argmax_guesses = error_labels_argmax_guess[error_type][layer][head_index]
                assert(len(error_labels) == len(error_values))
                # compute relative to errors - how much the error tokens attend to other tokens

                # 'balanced' accounts for the huge skew in 0 and 1 labels
                clf = LogisticRegression(random_state=0, class_weight='balanced')
                values_train, values_test, labels_train, labels_test = train_test_split(error_values, error_labels,
                                                                                        random_state=1337,
                                                                                        test_size=0.2)
                clf.fit([[x] for x in values_train], labels_train)
                predicted_labels_test = clf.predict([[x] for x in values_test])
                regression_f1 = f1_score(labels_test, predicted_labels_test)
                regression_results[error_type][layer].append(regression_f1)

                argmax_f1 = f1_score(argmax_guesses, error_labels)
                argmax_results[error_type][layer].append(argmax_f1)

                error_r, error_p_val,  = pearsonr(error_values, error_labels)
                pearson_results[error_type][layer].append((error_r, error_p_val))

    return pearson_results, regression_results, argmax_results


# prepended token count is the number of extra tokens added to the front (see: GECToR's $START token)
def attention_analysis(model: lit_model.Model, data: lit_dataset.Dataset, attention_heads: int, attention_layers: int,
                       max_len: int, prepended_token_count: int = 1, notebook=True):

    results = _get_results_from_model(model, data, notebook)
    return _perform_analysis(results, data, attention_heads, attention_layers, max_len, prepended_token_count, notebook)


if __name__ == '__main__':
    model = GectorBertModel('bert_0_gector.th')
    data = GeceProdigyData('test_sample.jsonl', gece_tags=True)
    pearson, regression, argmax = attention_analysis(model, data, model.ATTENTION_HEADS, model.ATTENTION_LAYERS,
                                                     model.MAX_LEN)
    print('debug')
