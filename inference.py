#IMPORTANTE
# // Este script de inferencia mezcla varios conceptos de archivos anteriores
#// pero fue hecho con ayuda de inteligencia artificial para poder comprobar 
#// que el modelo entrenado funciona correctamente.
#// Se tomo esta decision porque las GPU de colab estan un poco costosas y teniamos
#// el tiempo contado para poder entrenar el modelo y hacer la inferencia.
#// De este proyecto fue el unico codigo hecho con ayuda de inteligencia artificial, el resto del proyecto fue hecho por nosotros mismos.
#// 
#//
#// Charbel Ochoa 2026





import torch
from pathlib import Path
from tokenizers import Tokenizer
from model import build_transformer
from dataset import causal_mask
from config import get_config


config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_en = Tokenizer.from_file("tokenizer_en.json")
tokenizer_es = Tokenizer.from_file("tokenizer_es.json")

model = build_transformer(
    tokenizer_en.get_vocab_size(),
    tokenizer_es.get_vocab_size(),
    config["seq_len"],
    config["seq_len"],
    d_model=config["d_model"]
).to(device)

checkpoint_path = sorted(Path("weights").glob("tmodel_*.pt"))[-1]               #El ultimo checkpoint guardado
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Checkpoint:", checkpoint_path)



frase = input("Frase a traducir en inglés: ")

sos_en = tokenizer_en.token_to_id("[SOS]")
eos_en = tokenizer_en.token_to_id("[EOS]")
sos_es = tokenizer_es.token_to_id("[SOS]")
eos_es = tokenizer_es.token_to_id("[EOS]")

src_ids = [sos_en] + tokenizer_en.encode(frase).ids + [eos_en]

src = torch.tensor([src_ids], dtype=torch.long, device=device)
src_mask = torch.ones(1, 1, 1, src.size(1), dtype=torch.bool, device=device)

with torch.inference_mode():
    encoder_output = model.encode(src, src_mask)

    output = torch.tensor([[sos_es]], dtype=torch.long, device=device)

    for _ in range(config["seq_len"] - 1):
        tgt_mask = causal_mask(output.size(1)).unsqueeze(1).to(device)

        decoder_output = model.decode(
            encoder_output, src_mask, output, tgt_mask
        )

        logits = model.project(decoder_output[:, -1])
        next_token = logits.argmax(dim=-1).item()

        output = torch.cat([
            output,
            torch.tensor([[next_token]], device=device)
        ], dim=1)

        if next_token == eos_es:
            break

traduccion = tokenizer_es.decode(output[0].tolist())


print("Inglés:", frase)
print("Español:", traduccion)
