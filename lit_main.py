from lit_nlp import dev_server
from lit_nlp import server_flags
from absl import app

from lit_classes import GectorBertModel, GeceProdigyData


def main(_):
    models = {"gector": GectorBertModel('bert_0_gector.th')}
    datasets = {"test_data": GeceProdigyData('test_sample.jsonl')}

    # Start the LIT server. See server_flags.py for server options.
    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    lit_demo.serve()


if __name__ == "__main__":
    app.run(main)
