from pathlib import Path

import pandas as pd
import evaluate
from tqdm import tqdm

cer_metric = evaluate.load("cer")

import torch
from torch.utils.data import Dataset, DataLoader
from datautils import TibetanImageLinePairDataset
from transformers import VisionEncoderDecoderModel, PreTrainedModel

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import os


def ddp_setup(rank, world_size):
    """

    :param rank:
    :param world_size:
    :return:
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

class Trainer:
    def __init__(
            self,
            model: PreTrainedModel,
            train_data: DataLoader,
            eval_data: DataLoader,
            processor,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,
            # snapshot_path: str
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.eval_data = eval_data
        self.optimizer = optimizer
        self.processor = processor
        self.save_every = save_every
        self.epoch_run = 0
        # self.snapshot_path = snapshot_path
        # if os.path.exists(snapshot_path):
        #     print("Loading snapshot...")
        #     self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, batch):
        outputs = self.model(**batch)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def _evaluate(self, batch):
        outputs = self.model.generate(batch["pixel_values"])
        cer = self._compute_cer(pred_ids=outputs, label_ids=batch["labels"])
        return cer

    def _run_epoch(self, epoch):
        b_sz = 4
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.model.train()
        train_loss = 0.0
        for batch in tqdm(self.train_data):
            batch = {k: v.to(self.gpu_id) for k, v in batch.items()}
            loss = self._run_batch(batch)
            train_loss += loss

        print(f"Loss after epoch {epoch}:", train_loss / len(self.train_data))

        # evaluate
        self.model.eval()
        valid_cer = 0.0
        with torch.no_grad():
            for batch in tqdm(self.eval_data):
                batch = {k: v.to(self.gpu_id) for k, v in batch.items()}
                cer = self._evaluate(batch)
                valid_cer += cer

        current_cer = valid_cer / len(self.train_data)
        print(f"Validation CER after epoch {epoch}:", current_cer)

    def _compute_cer(self, pred_ids, label_ids):
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return cer

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def create_dataframe_from_data():
    from tqdm import tqdm
    # get all the file names
    file_paths = [file_path for file_path in list(Path('./tibetan-dataset/transcript/').iterdir())
                  if file_path.suffix == '.csv']
    file_paths = file_paths[:2]

    dfs = []

    for file_path in tqdm(file_paths):
        batch_name = file_path.name.removesuffix('.csv')
        df = pd.read_csv(str(file_path), sep=',')
        df['batch_name'] = batch_name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # change the column name line_image_id to file_name
    df.rename(columns={'line_image_id': 'file_name'}, inplace=True)

    # remove the rows that their image file does not exist
    df = df[~df['file_name'].isin(['I4PD45640075_1.jpg', 'I4PD45640075_2.jpg', 'I4PD45640075_8.jpg', 'I4PD45640075_9.jpg'])]

    print("Removed rows with missing image files.")

    return df


def split_train_test(df):
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(df, test_size=0.2)
    # we reset the indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return train_df, test_df


def load_train_objs():
    from transformers import TrOCRProcessor, ViTImageProcessor, RobertaTokenizer

    encode, decode = "google/vit-base-patch16-224-in21k", "sangjeedondrub/tibetan-roberta-base"

    feature_extractor = ViTImageProcessor.from_pretrained(encode)
    tokenizer = RobertaTokenizer.from_pretrained(decode)
    processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

    df = create_dataframe_from_data()
    train_df, test_df = split_train_test(df)

    train_set = TibetanImageLinePairDataset(root_dir='./tibetan-dataset/train/',
                                            df=train_df,
                                            processor=processor)
    eval_set = TibetanImageLinePairDataset(root_dir='./tibetan-dataset/train/',
                                           df=test_df,
                                           processor=processor)
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encode, decode)

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 512
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.5
    model.config.num_beams = 4

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    return train_set, eval_set, processor, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, total_epochs, save_every, batch_size):
    ddp_setup(rank, world_size)
    train_dataset, eval_dataset, processor, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(train_dataset, batch_size)
    eval_data = prepare_dataloader(eval_dataset, batch_size, shuffle=False)
    trainer = Trainer(model, train_data, eval_data, processor, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=4, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.total_epochs, args.save_every, args.batch_size), nprocs=world_size)
