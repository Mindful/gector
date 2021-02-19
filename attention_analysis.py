from lit_classes import Bea2019Data, GectorBertModel, \
    GECE_ERROR_TYPES  # reuse the LIT classes as they're already packaged for analysis

from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np


def main():
    model = GectorBertModel('bert_0_gector.th')
    data = Bea2019Data('data/test.jsonl', gece_tags=True)

    batch_size = model.max_minibatch_size()
    results = []
    for i in range(0, len(data), batch_size):
        batch_examples = data.examples[i:i + batch_size]
        results.extend(model.predict_minibatch(batch_examples))

    layers = ['layer{}'.format(x) for x in range(0, model.ATTENTION_LAYERS)] + ['layer_average']
    error_token_labels = {et: [] for et in GECE_ERROR_TYPES}
    error_token_values = {et: {layer: [[] for x in range(0, model.ATTENTION_LAYERS)] for layer in layers}
                          for et in GECE_ERROR_TYPES}
    error_labels_argmax_guess = {et: {layer: [[] for x in range(0, model.ATTENTION_LAYERS)] for layer in layers}
                                 for et in GECE_ERROR_TYPES}

    for result_dict, input_dict in zip(results, data.examples):
        input_tokens = input_dict['input_tokens']
        processed_input_tokens = result_dict['input_tokens']
        assert (input_tokens == processed_input_tokens[1:])
        if len(input_tokens) + 1 >= result_dict[layers[0]].shape[1]:
            continue  # input was truncated, don't bother with it
        for marking in input_dict['markings']:
            error_type = marking['error_type']
            labels_relative_to_error = [1 if idx in marking['cause_indices'] else 0 for idx in range(len(input_tokens))]

            for error_token_index in marking['error_indices']:
                error_token_labels[error_type].extend(labels_relative_to_error)
                for layer in layers:
                    for head_index in range(0, model.ATTENTION_HEADS):
                        attention_for_tokens = result_dict[layer][head_index, error_token_index,
                                                                 range(1, (len(input_tokens) + 1))]
                        max_attention_index = np.argmax(attention_for_tokens)
                        argmax_label_guess = [1 if idx == max_attention_index else 0 for idx
                                              in range(len(attention_for_tokens))]
                        error_token_values[error_type][layer][head_index].extend(attention_for_tokens)
                        error_labels_argmax_guess[error_type][layer][head_index].extend(argmax_label_guess)

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

    error_significant_seeming = 0
    for error_type in GECE_ERROR_TYPES:
        error_labels = error_token_labels[error_type]
        if len(error_labels) < 10:
            continue  # not enough data to compute anything for this type
        for layer in layers:
            for head_index in range(0, model.ATTENTION_HEADS):
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
                if error_p_val < 0.025:
                    error_significant_seeming += 1




    print('debug')


if __name__ == '__main__':
    main()
