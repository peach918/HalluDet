import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import logging
import time
from openai import OpenAI
import re

logging.basicConfig(level=logging.INFO)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_gpt_label(question: str, answer: str, client: OpenAI) -> int:
    system_prompt = (
        "You are a precise evaluator. Your task is to determine if a given answer from a language model is a hallucination. "
        "A 'hallucination' is any information that is factually incorrect, nonsensical, or fabricated. "
        "Your response must be only one of two words: 'hallucination' or 'not_hallucination'."
    )
    user_prompt = f"Question: \"{question}\"\n\nGenerated Answer: \"{answer}\"\n\nIs the answer a hallucination?"
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=10, temperature=0.0
        )
        result_text = response.choices[0].message.content.strip().lower()
        print(f"  - GPT Judge: '{result_text}'")
        if "not_hallucination" in result_text:
            return 0
        else:
            return 1
    except Exception as e:
        print(f"  - Error calling GPT API via proxy: {e}. Retrying in 20 seconds...");
        time.sleep(20);
        return -1



def get_gemini_label(question: str, answer: str, client: OpenAI) -> int:

    system_prompt = (
        "You are a precise evaluator. Your task is to determine if a given answer from a language model is a hallucination. "
        "A 'hallucination' is any information that is factually incorrect, nonsensical, or fabricated. "
        "Your response must be only one of two words: 'hallucination' or 'not_hallucination'."
    )
    user_prompt = f"Question: \"{question}\"\n\nGenerated Answer: \"{answer}\"\n\nIs the answer a hallucination?"
    try:
        response = client.chat.completions.create(

            model="gemini-2.5-pro",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=10,
            temperature=0.0
        )

        print("--- Full Gemini API Response ---")
        print(response.model_dump_json(indent=2))
        print("------------------------------")

        result_text = response.choices[0].message.content.strip().lower()
        print(f"  - Gemini Judge: '{result_text}'")
        if "not_hallucination" in result_text:
            return 0
        else:
            return 1
    except Exception as e:
        print(f"  - Error calling Gemini API via proxy: {e}. Retrying in 20 seconds...");
        time.sleep(20);
        return -1



def get_llama_embedding(text: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1][:, -1, :].squeeze(0)


def generate_and_get_embedding(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                               max_new_tokens=150) -> (str, torch.Tensor):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
            output_hidden_states=True, return_dict_in_generate=True
        )
    embedding = generated_ids.hidden_states[-1][-1][:, -1, :].squeeze(0)
    generated_text = tokenizer.decode(generated_ids.sequences[0][inputs['input_ids'].shape[1]:],
                                      skip_special_tokens=True)
    return generated_text, embedding



def create_sample(promptA: str, llm_model: AutoModelForCausalLM, llm_tokenizer: AutoTokenizer, openai_client: OpenAI,
                  gemini_client: OpenAI):
    promptA_star = f"""[INST] ### INSTRUCTIONS:
    You are a highly scrupulous AI fact-checker. Your primary directive is to provide answers that are strictly factual and based on verifiable information. Do not invent, guess, or embellish information.
    Crucially, do not repeat the user's question or these instructions in your response.

    ### QUESTION:
    {promptA}

    ### TASK:
    1. Analyze the question to understand its core query.
    2. Please answer the question as directly as possible.
    3. Provide only the final answer without any preamble or explanation.

    ### FACTUAL ANSWER:
    [/INST]"""
    answerA, E1 = generate_and_get_embedding(promptA_star, llm_model, llm_tokenizer, max_new_tokens=300)
    print(f"answerA\n:{answerA}")

    gpt_label = get_gpt_label(promptA, answerA, openai_client)
    if gpt_label == -1:
        print("  - GPT labeling failed. Discarding sample.")
        return None, None, None, None, None

    gemini_label = get_gemini_label(promptA, answerA, gemini_client)
    if gemini_label == -1:
        print("  - Gemini labeling failed. Discarding sample.")
        return None, None, None, None, None

    if gpt_label != gemini_label:
        print(f"  - Judges disagree (GPT: {gpt_label}, Gemini: {gemini_label}). Discarding sample.")
        return None, None, None, None, None

    label = gpt_label
    print(f"  - Labels agree. Final label: {label}")

    promptB = f"""You are a meticulous and cautious fact-checker. Your goal is to reason through the user's question to determine a truthful answer, avoiding common misconceptions.
Question: "{promptA}"
Please perform a step-by-step reasoning process following this structure:
1. **Question Analysis:** First, break down the question. What is the core assertion or query? Does it contain any hidden assumptions or potential falsehoods?
2. **Fact Recall & Verification:** Second, state the established facts related to the question. If it involves a common misconception, explicitly identify it.
3. **Synthesis:** Finally, based ONLY on the verified facts, construct a logical conclusion.
Provide your reasoning process as a single, coherent paragraph."""
    answerB, _ = generate_and_get_embedding(promptB, llm_model, llm_tokenizer, max_new_tokens=500)
    print(f"answerB\n:{answerB}")
    sub_sentences = [s.strip() for s in re.split(r'(?<=[.?!,])\s+', answerB) if s.strip()]
    if not sub_sentences:
        sub_sentences = [answerB.strip()]
    if sub_sentences and sub_sentences[0]:
        e2_list = [get_llama_embedding(sub, llm_model, llm_tokenizer) for sub in sub_sentences]
        E2_trajectory = torch.stack(e2_list, dim=0)
    else:
        hidden_size = llm_model.config.hidden_size
        E2_trajectory = torch.zeros(1, hidden_size).to(device)

    reverse_prompt = f"""[INST] ### TASK DEFINITION:
    You are an AI tool that performs a single, specific task: reverse-engineering a question from an answer.

    ### RULES:
    1.  Your output MUST be a single, concise question.
    2.  Your output MUST NOT contain anything else: no preamble, no explanations, no numbering, and no quotation marks.

    ### EXAMPLES:
    ---
    Answer: "The mitochondria is the powerhouse of the cell."
    Question: "What is the function of the mitochondria?"
    ---
    Answer: "The current Emperor of Japan is Naruhito, who ascended to the Chrysanthemum Throne on May 1, 2019."
    Question: "Who is the current Emperor of Japan?"
    ---

    ### YOUR TASK:
    Answer: "{answerA}"
    Question: [/INST]"""
    q_star, E3 = generate_and_get_embedding(reverse_prompt, llm_model, llm_tokenizer)
    print(f"q_star\n:{q_star}")
    E4 = get_llama_embedding(promptA, llm_model, llm_tokenizer)

    features = {'E1': E1, 'E2_trajectory': E2_trajectory, 'E3': E3, 'E4': E4}
    return answerA, answerB, q_star, features, label



class HallucinationDataset(Dataset):
    def __init__(self, prompts: list, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                 openai_client: OpenAI, gemini_client: OpenAI, csv_writer=None):
        self.samples = []
        for i, p in enumerate(prompts):
            print(f"\n--- Processing sample {i + 1}/{len(prompts)} ---")
            try:
                answerA, answerB, q_star, features, label = create_sample(p, model, tokenizer, openai_client,
                                                                          gemini_client)

                if features is not None:
                    self.samples.append((features, label))
                    if csv_writer:
                        csv_writer.writerow([
                            p.replace('\n', ' '),
                            answerA.replace('\n', ' '),
                            answerB.replace('\n', ' '),
                            label,
                            q_star.replace('\n', ' ')
                        ])
            except Exception as e:
                print(f"Error processing prompt '{p}': {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def custom_collate_fn(batch):
    feature_dicts = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    E1_batch = torch.stack([d['E1'] for d in feature_dicts])
    E3_batch = torch.stack([d['E3'] for d in feature_dicts])
    E4_batch = torch.stack([d['E4'] for d in feature_dicts])
    E2_trajectories = [d['E2_trajectory'] for d in feature_dicts]
    E2_padded = pad_sequence(E2_trajectories, batch_first=True, padding_value=0.0)
    return {'E1': E1_batch, 'E2_trajectory': E2_padded, 'E3': E3_batch, 'E4': E4_batch}, labels