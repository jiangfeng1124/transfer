
from .lstm import LSTM
from .cnn import CNN
from .mlp import MLP
from .tagger import Tagger

MODEL_CLASS = {
    'lstm' : LSTM,
    'gru' : LSTM,
    'cnn' : CNN,
    'mlp' : MLP,
    'tagger': Tagger,
}

from .domain_critic import ClassificationD, MMD, CoralD, WassersteinD

CRITIC_CLASS = {
    'dann' : ClassificationD,
    'mmd' : MMD,
    'coral' : CoralD,
    'wd' : WassersteinD
}

def get_model_class(model_name):
    model_name = model_name.lower()
    model = MODEL_CLASS.get(model_name, None)
    if model is None:
        raise Exception("Unknown model class: {}".format(
            model_name
        ))
    return model

def get_critic_class(critic_name):
    critic_name = critic_name.lower()
    critic = CRITIC_CLASS.get(critic_name, None)
    if critic is None:
        raise Exception("Unknown critic class: {}".format(
            critic_name
        ))
    return critic

def get_decoder_model_class():
    model_name = "lstm_decoder"
    model = MODEL_CLASS.get(model_name, None)
    if model is None:
        raise Exception("Unknown model class: {}".format(
            model_name
        ))
    return model
