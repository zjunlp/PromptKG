from .base_data_module import BaseDataModule
from .processor import KGCDataset, CustomSampler, get_dataset, KGProcessor
from transformers import AutoTokenizer, BertTokenizer
from torch.utils.data import DataLoader





class KGC(BaseDataModule):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.processor = KGProcessor()
        self.label_list = self.processor.get_labels(args.data_dir)

    def setup(self, stage=None):
        self.data_train = get_dataset(self.args, self.processor, self.label_list, self.tokenizer, "train")
        self.data_val = get_dataset(self.args, self.processor, self.label_list, self.tokenizer, "dev")
        self.data_test = get_dataset(self.args, self.processor, self.label_list, self.tokenizer, "test")

    def prepare_data(self):
        pass


    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--data_dir", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=256, help="Number of examples to operate on per forward step.")
        parser.add_argument("--few", type=int, default=5, help="Number of examples to operate on per forward step.")
        parser.add_argument("--warm_up_radio", type=float, default=0.1, help="Number of examples to operate on per forward step.")
        parser.add_argument("--eval_batch_size", type=int, default=5000)
        parser.add_argument("--overwrite_cache", action="store_true", default=False)
        return parser

    def get_tokenizer(self):
        return self.tokenizer

    # def train_dataloader(self):
    #     return DataLoader(self.data_train, num_workers=self.num_workers, pin_memory=True, batch_sampler=self.sampler_train)

    # def val_dataloader(self):
    #     return DataLoader(self.data_val, num_workers=self.num_workers, pin_memory=True, batch_sampler=self.sampler_val)

    # def test_dataloader(self):
    #     return DataLoader(self.data_test, num_workers=self.num_workers, pin_memory=True, batch_sampler=self.sampler_test)