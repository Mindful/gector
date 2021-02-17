from lit_classes import Bea2019Data, GectorBertModel, \
    GECE_ERROR_TYPES  # reuse the LIT classes as they're already packaged for analysis

from scipy.stats import pearsonr

def main():
    model = GectorBertModel('bert_0_gector.th')
    data = Bea2019Data('data/test.json', gece_tags=True)

    batch_size = model.max_minibatch_size()
    results = []
    for i in range(0, len(data), batch_size):
        batch_examples = data.examples[i:i + batch_size]
        results.extend(model.predict_minibatch(batch_examples))

    layers = ['layer{}'.format(x) for x in range(0, model.ATTENTION_LAYERS)] + ['layer_average']
    error_token_labels = {et: [] for et in GECE_ERROR_TYPES}
    cause_token_labels = {et: [] for et in GECE_ERROR_TYPES}
    error_token_values = {et: {layer: [[] for x in range(0, model.ATTENTION_LAYERS)] for layer in layers}
                          for et in GECE_ERROR_TYPES}
    cause_token_values = {et: {layer: [[] for x in range(0, model.ATTENTION_LAYERS)] for layer in layers}
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
            labels_relative_to_cause = [1 if idx in marking['error_indices'] else 0 for idx in range(len(input_tokens))]

            for error_token_index in marking['error_indices']:
                error_token_labels[error_type].extend(labels_relative_to_error)
                for layer in layers:
                    for head_index in range(0, model.ATTENTION_HEADS):
                        attention_for_tokens = result_dict[layer][head_index, error_token_index,
                                                                 range(1, (len(input_tokens) + 1))]
                        error_token_values[error_type][layer][head_index].extend(attention_for_tokens)

            for cause_token_index in marking['cause_indices']:
                cause_token_labels[error_type].extend(labels_relative_to_cause)
                for layer in layers:
                    for head_index in range(0, model.ATTENTION_HEADS):
                        attention_for_tokens = result_dict[layer][head_index, cause_token_index,
                                                                 range(1, (len(input_tokens) + 1))]
                        cause_token_values[error_type][layer][head_index].extend(attention_for_tokens)

    pearson_error_results = {
        error_type: {
            layer: [] for layer in layers
        } for error_type in GECE_ERROR_TYPES
    }
    pearson_cause_results = {
        error_type: {
            layer: [] for layer in layers
        } for error_type in GECE_ERROR_TYPES
    }
    error_significant_seeming = 0
    cause_significant_seeming = 0
    for error_type in GECE_ERROR_TYPES:
        error_labels = error_token_labels[error_type]
        cause_labels = cause_token_labels[error_type]
        if len(error_labels) < 10 or len(cause_labels) < 10:
            continue  # not enough data to compute anything for this type
        for layer in layers:
            for head_index in range(0, model.ATTENTION_HEADS):
                error_values = error_token_values[error_type][layer][head_index]
                cause_values = cause_token_values[error_type][layer][head_index]
                assert(len(error_labels) == len(error_values))
                assert(len(cause_labels) == len(cause_values))
                # compute relative to errors - how much the error tokens attend to other tokens
                error_r, error_p_val,  = pearsonr(error_labels, error_values)
                pearson_error_results[error_type][layer].append((error_r, error_p_val))
                if error_p_val < 0.05:
                    error_significant_seeming += 1

                # compute relative to causes - how much the cause tokens attend to other tokens
                cause_r, cause_p_val = pearsonr(cause_labels, cause_values)
                pearson_cause_results[error_type][layer].append((cause_r, cause_p_val))
                if cause_p_val < 0.05:
                    cause_significant_seeming += 1


    print('debug')


if __name__ == '__main__':
    main()
