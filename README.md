# Relevance-Paraphrasing

## Environment

```
pip install -r requirements.txt

```
## Paraphrasing Articles

Generate Paraphrased Articles via running the following command in the llama2 directory. See Llama section for model downloading details.

```
torchrun --nproc_per_node 2  --nproc_per_node 2 create_paraphrase.py \ --ckpt_dir llama-2-13b-chat/ \ --tokenizer_path tokenizer.model \ --max_seq_len 2000 --max_batch_size 4

```

## Dolly-v2-7b

Generate Original Summaries with the command

```
python inference_original.py

```

Generate Paraphrased Summaries with the command

```
python inference_paraphrase.py

```

Generate results via script_original.py and script_paraphrased.py

Commands to run scripts:

```
python script_original.py --dataset dataset_name 

```
Replace dataset_name with either cnn, xsum, news, reddit


```
python script_paraphrased.py --dataset dataset_name 

```
Replace dataset_name with either cnn, xsum, news, reddit 



## ChatGPT 3.5-T

Inference the LLM with running inference.ipynb for original summaries

Generate Paraphrased Summaries with the command

```
python inference_paraphrase.py

```

You will need to add OpenAI API key in a .env file.


Generate results via script_original.py and script_paraphrased.py

Commands to run scripts:

```
python script_original.py --dataset dataset_name 

```
Replace dataset_name with either cnn, xsum, news, reddit


```
python script_paraphrased.py --dataset dataset_name 

```
Replace dataset_name with either cnn, xsum, news, reddit 


## Llama-13b-chat

You will need to download and copy the Llama folder from meta with the weights to the directory first.
Inference can be done by running the files inference_original.py and inference_paraphrase.py with the following commands.

```
torchrun --nproc_per_node 2 inference_original.py \
    --ckpt_dir llama-2-13b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 2000 --max_batch_size 4

```

```
torchrun --nproc_per_node 2 inference_paraphrase.py \
    --ckpt_dir llama-2-13b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 2000 --max_batch_size 4

```


Generate results via script_original.py and script_paraphrased.py

Commands to run scripts:

```
python script_original.py --dataset dataset_name 

```
Replace dataset_name with either cnn, xsum, news, reddit


```
python script_paraphrased.py --dataset dataset_name 

```
Replace dataset_name with either cnn, xsum, news, reddit 

## Mistral-7B

Generate Original Summaries with the command

```
python inference_original.py

```

Generate Paraphrased Summaries with the command

```
python inference_paraphrase.py

```

Generate results via script_original.py and script_paraphrased.py

Commands to run scripts:

```
python script_original.py --dataset dataset_name 

```
Replace dataset_name with either cnn, xsum, news, reddit


```
python script_paraphrased.py --dataset dataset_name 

```
Replace dataset_name with either cnn, xsum, news, reddit 
