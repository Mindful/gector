from lit_nlp.api import model as lit_model
from typing import Any, List
from gector.gec_model import GecBERTModel
from lit_nlp.api import types as lit_types
from lit_nlp.api import dataset as lit_dataset
from lit_nlp import dev_server
from lit_nlp import server_flags
from absl import app
import numpy

class Bea2019Data(lit_dataset.Dataset):

    def __init__(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()

        self._examples = [{'input_text': x.strip(), 'target_text': 'I like dogs.'} for x in lines if x.strip()]

    def spec(self) -> lit_types.Spec:
        """Should match MLM's input_spec()."""
        return {'input_text': lit_types.TextSegment(),
                'target_Text': lit_types.TextSegment()}


class GectorBertModel(lit_model.Model):
    ATTENTION_LAYERS = 12

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
            "target_text": lit_types.TextSegment(required=False)
        }

    def output_spec(self) -> lit_types.Spec:
        output = {"input_tokens": lit_types.Tokens(parent="input_text"),
                  "predicted": lit_types.GeneratedText(parent='target_text'),
                  'layer_average': lit_types.AttentionHeads(align=('input_tokens', 'input_tokens'))}
        for layer in range(self.ATTENTION_LAYERS):
            output['layer{}'.format(layer)] = lit_types.AttentionHeads(align=('input_tokens', 'input_tokens'))

        return output


def main(_):
    models = {"gector": GectorBertModel('bert_0_gector.th')}
    datasets = {"test_data": Bea2019Data('data/head_test.txt')}

    # Start the LIT server. See server_flags.py for server options.
    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    lit_demo.serve()


if __name__ == "__main__":
    app.run(main)
