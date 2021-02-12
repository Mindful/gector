from lit_nlp.api import model as lit_model
from typing import Any, List
from gector.gec_model import GecBERTModel
from lit_nlp.api import types as lit_types
from lit_nlp.api import dataset as lit_dataset
from lit_nlp import dev_server
from lit_nlp import server_flags
from absl import app


class Bea2019Data(lit_dataset.Dataset):

    def __init__(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()

        self._examples = [{'original': x.strip()} for x in lines if x.strip()]

    def spec(self) -> lit_types.Spec:
        """Should match MLM's input_spec()."""
        return {'original': lit_types.TextSegment()}


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
        # this breaks down if we have duplicates, but we shouldn't
        sentence_indices = [(ex["original"], index) for index, ex in enumerate(inputs)]
        tokenized_input_with_indices = [(original.split(), index) for original, index in sentence_indices]
        batch = [(tokens, index) for tokens, index in tokenized_input_with_indices
                 if len(tokens) >= self.model.min_len]

        # anything under min length doesn't get processed anyway, so we don't pass it in and just keep it
        # so we can put stuff back in order later
        ignored = [(tokens, index) for tokens, index in tokenized_input_with_indices
                   if len(tokens) < self.model.min_len]

        model_input = [tokens for tokens, index in batch]
        preds, _, attention = self.model.handle_batch(model_input)
        attention = attention[0]  # we only have one iteration

        assert (len(preds) == len(attention))
        output = [{'tokens': x} for x in preds]

        batch_iter = iter(batch)
        for output_dict, attention_info in zip(output, attention):
            original_tokens, original_index = next(batch_iter)
            output_dict['original_index'] = original_index
            for layer_index, attention_layer_info in enumerate(attention_info):
                output_dict['attention_layer{}'.format(layer_index)] = attention_layer_info

        output.extend({'tokens': tokens, 'original_index': original_index} for tokens, original_index in ignored)
        output.sort(key=lambda x: x['original_index'])
        for d in output:
            del d['original_index']
        return output

    def input_spec(self) -> lit_types.Spec:
        return {
            "original": lit_types.TextSegment(),
        }

    def output_spec(self) -> lit_types.Spec:
        output = {
            "tokens": lit_types.Tokens(),
        }

        for layer in range(self.ATTENTION_LAYERS):
            output['attention_layer{}'.format(layer)] = lit_types.AttentionHeads(align=('tokens', 'tokens'))

        return output


def main(_):
    models = {"gector": GectorBertModel('bert_0_gector.th')}
    datasets = {"test_data": Bea2019Data('data/head_test.txt')}

    # Start the LIT server. See server_flags.py for server options.
    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    lit_demo.serve()


if __name__ == "__main__":
    app.run(main)
