#/---- Loop de Entrenamiento ----/
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
from config import get_config, get_weights_file_path, latest_weights_file_path

from dataset import BillingualDataset, causal_mask

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenzier(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
    

def get_ds(config):
    ds_raw = load_dataset("Helsinki-NLP/opus_books", f'{config["lang_src"]}-{config["lang_tgt"]}', split="train")

    #Build tokenizer

    tokenizer_src = get_or_build_tokenzier(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenzier(config, ds_raw, config['lang_tgt'])

        
    ds_raw = ds_raw.filter(
        lambda item:
            len(tokenizer_src.encode(item["translation"][config["lang_src"]]).ids) <= config["seq_len"] - 2
            and len(tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids) <= config["seq_len"] - 1
    )



    # Keep 90% of the dataset for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])


    train_ds = BillingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BillingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])


    max_len_src = 0 
    max_len_tgt = 0 

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))


    print(f'Tamaño maximo para de frase de entrada: {max_len_src}')
    print(f'Tamaño maximo para de frase de target: {max_len_tgt}')


    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)



    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt



def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'],config['seq_len'], d_model=config['d_model'])
    return model



def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (f'Usando dispositivo: {device}')


    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_data_loader, val_data_loader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])


    #AdamW optimizer con PyTorch
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    initial_epoch = 0
    global_step = 0
    preload = config['preload']

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Cargando pesos desde: {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
        optimizer.load_state_dict(state['optimizer_state_dict'])


    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device)


    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_data_loader, desc=f'Epoch {epoch: 02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)


            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)



            label = batch['label'].to(device)


            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f'loss': f'{loss.item():6.3f}'})


            #Log loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            #Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1


        #Save model weights
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
