from typing import Any, List
import numpy
from lit_nlp.api import model as lit_model, types as lit_types, dataset as lit_dataset
import jsonlines
from gector.gec_model import GecBERTModel


class GectorBertModel(lit_model.Model):
    ATTENTION_LAYERS = 12
    ATTENTION_HEADS = 12

    def __init__(self, model_path):
        self.model = GecBERTModel(vocab_path='data/output_vocabulary',
                                  model_paths=[model_path],
                                  max_len=50,
                                  min_len=3,
                                  iterations=1,  # limit to 1 iteration to make attention analysis reasonable
                                  min_error_probability=0.0,
                                  model_name='bert',  # we're using BERT
                                  special_tokens_fix=0,  # disabled for BERT
                                  log=True,
                                  confidence=0,
                                  is_ensemble=0,
                                  weigths=None)

    # LIT API implementation
    def max_minibatch_size(self, config: Any = None) -> int:
        return 32

    def predict_minibatch(self, inputs: List, config=None) -> List:
        # we append '$START' to the beginning of token lists because GECTOR does as well (and this is what BERT
        # ends up processing. see preprocess() and postprocess_batch() in gec_model

        # this breaks down if we have duplicates, but we shouldn't
        sentence_indices = [(ex["input_text"], index) for index, ex in enumerate(inputs)]
        tokenized_input_with_indices = [(original.split(), index) for original, index in sentence_indices]
        batch = [(tokens, index) for tokens, index in tokenized_input_with_indices
                 if len(tokens) >= self.model.min_len]

        # anything under min length doesn't get processed anyway, so we don't pass it in and just keep it
        # so we can put stuff back in order later
        ignored = [(tokens, index) for tokens, index in tokenized_input_with_indices
                   if len(tokens) < self.model.min_len]

        model_input = [tokens for tokens, index in batch]
        predictions, _, attention = self.model.handle_batch(model_input)
        attention = attention[0]  # we only have one iteration

        assert (len(predictions) == len(attention))
        output = [{'predicted': ' '.join(tokenlist)} for tokenlist in predictions]

        # wanted to average across heads and layers, but attention with different head counts breaks LIT
        #attention_averaged = numpy.average(attention, (1, 2))[:, numpy.newaxis, ...]

        batch_iter = iter(batch)
        for output_dict, attention_info in zip(output, attention):
            original_tokens, original_index = next(batch_iter)
            output_dict['original_index'] = original_index
            output_dict['layer_average'] = numpy.average(attention_info, axis=0)

            for layer_index, attention_layer_info in enumerate(attention_info):
                output_dict['layer{}'.format(layer_index)] = attention_layer_info

        output.extend({'predicted': ' '.join(tokens), 'original_index': original_index} for tokens, original_index in ignored)
        output.sort(key=lambda x: x['original_index'])
        for tokenized_input, index in tokenized_input_with_indices:
            output[index]['input_tokens'] = ['$START'] + tokenized_input

        for d in output:
            del d['original_index']

        return output

    def input_spec(self) -> lit_types.Spec:
        return {
            "input_text": lit_types.TextSegment(),
            "target_text": lit_types.TextSegment()
        }

    def output_spec(self) -> lit_types.Spec:
        output = {"input_tokens": lit_types.Tokens(parent="input_text"),
                  "predicted": lit_types.GeneratedText(parent='target_text'),
                  'layer_average': lit_types.AttentionHeads(align=('input_tokens', 'input_tokens'))}
        for layer in range(self.ATTENTION_LAYERS):
            output['layer{}'.format(layer)] = lit_types.AttentionHeads(align=('input_tokens', 'input_tokens'))

        return output


AGREEMENT_PERSON = 'AGREEMENT_PERSON'
AGREEMENT_PLURAL = 'AGREEMENT_PLURAL'
AGREEMENT_TENSE = 'AGREEMENT_TENSE'
GECE_ERROR_TYPES = {AGREEMENT_PERSON, AGREEMENT_TENSE, AGREEMENT_PLURAL}


class Bea2019Data(lit_dataset.Dataset):

    NONE_TAG = 'O'

    def __init__(self, path, gece_tags=False):
        with jsonlines.open(path) as lines:
            self._examples = []
            for jsonline in lines:
                input_text = jsonline['source_text']
                result = {'input_text': input_text, 'input_tokens': input_text.split(),
                          'target_text': jsonline['target_text']}

                if gece_tags and 'markings' in jsonline:
                    markings = jsonline['markings']
                    result['markings'] = markings  # not in the output spec because only used in attention analysis (not LIT)
                    tags = [Bea2019Data.NONE_TAG] * len(input_text)
                    for mark in markings:
                        mark_type = mark['error_type']
                        for idx in mark['error_indices']:
                            tags[idx] = '{}:ERR'.format(mark_type)

                        for idx in mark['cause_indices']:
                            tags[idx] = '{}:SRC'.format(mark_type)
                    result['gece_tags'] = tags

                self.examples.append(result)

    def spec(self) -> lit_types.Spec:
        """Should match MLM's input_spec()."""
        return {'input_text': lit_types.TextSegment(),
                'target_text': lit_types.TextSegment(),
                'input_tokens': lit_types.Tokens(required=False),
                'gece_tags': lit_types.SequenceTags(align='input_tokens', required=False)}
