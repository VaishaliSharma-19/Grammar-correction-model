# Grammar-correction-model
This project focuses on implementing a grammar correction model using the GPT-2 language model. The model is fine-tuned on a small dataset taken from Hugging Face containing input sentences and their corresponding corrected versions. The trained model can then be used to correct grammar in new sentences.

## To use the grammar correction model, follow the steps below:

- Clone the repository to your local machine:
  
      clone https://github.com/your_username/grammar-correction.git
      cd grammar-correction

- Set up the environment by installing the required libraries:
  
      !pip install datasets
      !pip install transformers
      !pip install torch
      !pip install pandas
      !pip install seaborn
      !pip install matplotlib
      !pip install nltk

- Load the dataset and preprocess it for training:

      from google.colab import drive
      drive.mount('/content/drive')
      
      from datasets import DatasetDict, load_dataset
      from sklearn.model_selection import train_test_split

- Load the dataset from the Hugging Face Datasets library:
  
      dataset = load_dataset("Owishiboo/grammar-correction")

- Split the dataset into train, test, and validation set:
      
      train_test_dataset = dataset['train'].train_test_split(test_size=0.1, shuffle=True)
      test_val_dataset = train_test_dataset['test'].train_test_split(test_size=0.5, shuffle=True)

- Combine the datasets into train, test, and validation DatasetDict:
  
      train_test_valid_dataset = DatasetDict({
          'train': train_test_dataset['train'],
          'test': test_val_dataset['train'],
          'valid': test_val_dataset['test']
      })

- Define the GPT2Dataset class for tokenization and data loading:
  
      import torch
      from transformers import GPT2Tokenizer
      
      class GPT2Dataset(torch.utils.data.Dataset):
          def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):
              # Code for tokenization and encoding of the input sentences
              ...
      
          def __len__(self):
              return len(self.input_ids)
      
          def __getitem__(self, idx):
              return self.input_ids[idx], self.attn_masks[idx]

- Load the GPT-2 tokenizer:

       tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='', eos_token='', pad_token='')

- Prepare the training and validation datasets:
  
       import pandas as pd

- Prepare the training and validation datasets for GPT2:
      
      train_texts = pd.Series(train_test_valid_dataset['train']['input']) + " Corrected: " + pd.Series(train_test_valid_dataset['train']['target'])
      val_texts = pd.Series(train_test_valid_dataset['valid']['input']) + " Corrected: " + pd.Series(train_test_valid_dataset['valid']['target'])

- Filter out non-string values from the training and validation datasets:
  
      train_texts_vals = [t for t in train_texts.values if isinstance(t, str)]
      val_texts_vals = [t for t in val_texts.values if isinstance(t, str)]

- Create GPT2Dataset objects for training and validation:
  
      train_dataset = GPT2Dataset(train_texts_vals, tokenizer, max_length=768)
      val_dataset = GPT2Dataset(val_texts_vals, tokenizer, max_length=768)

      batch_size = 1

- Create the DataLoaders for training and validation datasets:
      
      train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
      validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

- Load the pretrained GPT-2 model and fine-tune it on the training dataset:
    
      import os
      import time
      import datetime
      from transformers import GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup

- Load the GPT-2 model configuration:
  
      configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

- Instantiate the GPT-2 model:
  
      model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

- Resize the token embeddings to match the tokenizer:

      model.resize_token_embeddings(len(tokenizer))

- Set the device to run the model on GPU if available:
  
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      model.to(device)

- Define hyperparameters and optimizer:
  
      epochs = 2
      learning_rate = 5e-5
      warmup_steps = 1e2
      epsilon = 1e-8
      sample_every = 100
  
      optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
      total_steps = len(train_dataloader) * epochs
      scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

- Training and Validation Loop:
  
      training_stats = []
      total_t0 = time.time()
      for epoch_i in range(0, epochs):
  

- Save the fine-tuned model:
  
      model.save_pretrained("grammar-correction-model-gpt2")

- Load the fine-tuned model for grammar correction:
  
      model = GPT2LMHeadModel.from_pretrained('grammar-correction-model-gpt2')

- Use the predict function to correct grammar in a given sentence:

      def predict(txt):
              ids = tokenizer.encode(' ' + txt + ' Corrected:', return_tensors='pt')
              sample_outputs = model.generate(ids, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_k=50, max_length=100, top_p=0.95, num_return_sequences=1)
          
              for i, sample_output in enumerate(sample_outputs):
                  corrected_sent = tokenizer.decode(sample_output, skip_special_tokens=True)
                  print(f"Prompt: {txt}\n\nCorrected sentence generated: {corrected_sent}")
          
          predict("There is so many cars")

- Evaluating the Model:
  
    To evaluate the model's performance, the script contains a function GPT2score that calculates the perplexity score for a given sentence. The model's 
    performance can be assessed by generating corrected sentences and analyzing the perplexity scores.

      import numpy as np
      
      def GPT2score(sentence):
          tokenize_input = tokenizer.encode(sentence)
          tensor_input = torch.tensor([tokenize_input])
          loss = model(tensor_input, labels=tensor_input)[0]
          return np.exp(loss.detach().cpu().numpy())

- You can use the GPT2score function to evaluate the model's perplexity in different sentences.


## Conclusion:

The provided code allows you to fine-tune a GPT-2 model for grammar correction and use it to correct grammar in sentences. The model can be further evaluated using perplexity scores.

