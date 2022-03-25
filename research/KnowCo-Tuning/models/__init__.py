from .bert import BertForSequenceClassification
from .gpt2 import GPT2DoubleHeadsModel
from .gpt2.modeling_gpt2 import GPT2UseLabelWord

from .roberta import RobertaForSequenceClassification 
from .roberta.modeling_roberta import RobertaUseLabelWord


from .albert.modeling_albert import AlbertUseLabelWord, AlbertUseLabelWordForMRC
from .albert.modeling_albert import *


from .bert.modeling_bert import KGBERT,KGBERTUseLabelWord, KGBERTGetLabelWord