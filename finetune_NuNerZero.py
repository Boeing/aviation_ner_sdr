import json
from gliner import GLiNER
import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import os
from types import SimpleNamespace

torch.cuda.empty_cache()
print(f"Number of available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Current GPU device name: {torch.cuda.get_device_name(0)}")
print(f"Using device: {device}")
train_path = "./Dataset/BIO/train.10.23.bio.json" # your data or dataset from https://github.com/urchade/GLiNER/blob/main/examples/sample_data.json

with open(train_path, "r") as f:
    data = json.load(f)

# converting entities-level data to token-level data
new_data = []
for d in data:
    new_ner = []
    for s, f, c in d["ner"]:
        for i in range(s, f + 1):
            # labels are intended to be lower-case
            new_ner.append((i, i, c.lower()))
    new_d = {
        "tokenized_text": d["tokenized_text"],
        "ner": new_ner,
    }
    new_data.append(new_d)
data = new_data


def save_model(current_model, path):
    config = current_model.config
    dict_save = {"model_weights": current_model.state_dict(), "config": config}
    torch.save(dict_save, path)

def load_model(path, model_name=None):
    dict_load = torch.load(path, map_location=torch.device('cuda'))
    config = dict_load["config"]

    print(f"'{config.model_name}' should be available for local processing")

    if model_name is not None:
        config.model_name = model_name

    loaded_model = GLiNER(config)
    loaded_model.load_state_dict(dict_load["model_weights"])

    return loaded_model

# model = GLiNER.from_pretrained("nuZeroModel", local_files_only=True)
model = load_model("./nuZeroModel/pytorch_model.bin", model_name='nuZeroNER')

# Define the hyperparameters in a config variable
config = SimpleNamespace(
    num_steps=1000, # regualte number train, eval steps depending on the data size
    eval_every=500,
    train_batch_size=8, # regulate batch size depending on GPU memory available

    max_len=1024, # maximum sentence length, 2048 for NuNerZero_long_context
    save_directory="logs", # log dir
    device='cuda', # training device - cpu or cuda

    warmup_ratio=0.1, # keep other parameters unchanged
    lr_encoder=1e-5,
    lr_others=5e-5,
    freeze_token_rep=False,

    max_types=25,
    shuffle_types=True,
    random_drop=True,
    max_neg_type_ratio=1,
)

def train(model, config, train_data, eval_data=None):
    model = model.to(config.device)

    # Set sampling parameters from config
    model.set_sampling_params(
        max_types=config.max_types,
        shuffle_types=config.shuffle_types,
        random_drop=config.random_drop,
        max_neg_type_ratio=config.max_neg_type_ratio,
        max_len=config.max_len
    )

    model.train()

    # Initialize data loaders
    train_loader = model.create_dataloader(train_data, batch_size=config.train_batch_size, shuffle=True)

    # Optimizer
    optimizer = model.get_optimizer(config.lr_encoder, config.lr_others, config.freeze_token_rep)

    pbar = tqdm(range(config.num_steps))

    if config.warmup_ratio < 1:
        num_warmup_steps = int(config.num_steps * config.warmup_ratio)
    else:
        num_warmup_steps = int(config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=config.num_steps
    )

    iter_train_loader = iter(train_loader)

    for step in pbar:
        try:
            x = next(iter_train_loader)
        except StopIteration:
            iter_train_loader = iter(train_loader)
            x = next(iter_train_loader)

        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(config.device)

        loss = model(x)  # Forward pass

        # Check if loss is nan
        if torch.isnan(loss):
            continue

        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        scheduler.step()  # Update learning rate schedule
        optimizer.zero_grad()  # Reset gradients

        description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"
        pbar.set_description(description)

        if (step + 1) % config.eval_every == 0:

            model.eval()

            if eval_data is not None:
                results, f1 = model.evaluate(eval_data["samples"], flat_ner=True, threshold=0.5, batch_size=12,
                                     entity_types=eval_data["entity_types"])
                print(f"Step={step}\n{results}")

            if not os.path.exists(config.save_directory):
                os.makedirs(config.save_directory)

            model.save_pretrained(f"{config.save_directory}/finetuned_{step}")

            model.train()


eval_data = {
    "entity_types": ["B-prod", "I-prod", "B-loc", "I-loc", "B-pcon", "I-pcon", "B-sit", "I-sit", "B-act", "I-act", "B-bird", "I-bird", "B-flt", "I-flt"],
    "samples": data[:10]
}

train(model, config, data, eval_data)

model.save_pretrained("safety_NER_BIO")
