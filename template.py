from transformers import GenerationConfig

gsm8k_fewshot_prompt = """Question: In 2004, there were 60 kids at a cookout. In 2005, half the number of kids came to the cookout as compared to 2004. In 2006, 2/3 as many kids came to the cookout as in 2005. How many kids came to the cookout in 2006?
Let's think step by step
In 2005, 60/2=30 kids came to the cookout.
In 2006, 30/3*2=20 kids came to the cookout.
The answer is 20

Question: Zilla spent 7% of her monthly earnings on rent, half of it on her other monthly expenses, and put the rest in her savings. If she spent $133 on her rent, how much does she deposit into her savings account in a month?
Let's think step by step
Since $133 is equal to 7% of her earnings, then 1% is equal to $133/7 = $19.
The total monthly earning of Zilla is represented by 100%, so $19 x 100 = $1900 is her monthly earnings.
So, $1900/2 = $950 is spent on her other monthly expenses.
The total amount spent on the rent and other monthly expenses is $133 + $950 = $1083.
Hence, she saves $1900 - $1083 = $817 per month.
The answer is 817

Question: If Buzz bought a pizza with 78 slices at a restaurant and then decided to share it with the waiter in the ratio of 5:8, with Buzz's ratio being 5, what's twenty less the number of slices of pizza that the waiter ate?
Let's think step by step
The total ratio representing the slices of pizza that Buzz bought is 5+8=13
If he shared the slices of pizza with the waiter, the waiter received a fraction of 8/13 of the total number of slices, which totals 8/13 * 78 = 48 slices
Twenty less the number of slices of pizza that the waiter ate is 48-20 = 28
The answer is 28

Question: Jame gets a raise to $20 per hour and works 40 hours a week.  His old job was $16 an hour for 25 hours per week.  How much more money does he make per year in his new job than the old job if he works 52 weeks a year?
Let's think step by step
He makes 20*40=$800 per week
He used to make 16*25=$400 per week
So his raise was 800-400=$400 per week
So he makes 400*52=$20,800 per year more
The answer is 20800

Question: Mr. Gardner bakes 20 cookies, 25 cupcakes, and 35 brownies for his second-grade class of 20 students. If he wants to give each student an equal amount of sweet treats, how many sweet treats will each student receive?
Let's think step by step
Mr. Gardner bakes a total of 20 + 25 + 35 = 80 sweet treats
Each student will receive 80 / 20 = 4 sweet treats
The answer is 4

Question: A used car lot has 24 cars and motorcycles (in total) for sale. A third of the vehicles are motorcycles, and a quarter of the cars have a spare tire included. How many tires are on the used car lot’s vehicles in all?
Let's think step by step
The used car lot has 24 / 3 = 8 motorcycles with 2 tires each.
The lot has 24 - 8 = 16 cars for sale
There are 16 / 4 = 4 cars with a spare tire with 5 tires each.
The lot has 16 - 4 = 12 cars with 4 tires each.
Thus, the used car lot’s vehicles have 8 * 2 + 4 * 5 + 12 * 4 = 16 + 20 + 48 = 84 tires in all.
The answer is 84

Question: Norma takes her clothes to the laundry. She leaves 9 T-shirts and twice as many sweaters as T-shirts in the washer. When she returns she finds 3 sweaters and triple the number of T-shirts. How many items are missing?
Let's think step by step
Norma left 9 T-shirts And twice as many sweaters, she took 9 * 2= 18 sweaters
Adding the T-shirts and sweaters, Norma left 9 + 18 = 27 clothes
When she came back, she found 3 sweaters And triple the number of T-shirts, she found 3 * 3 = 9 T-shirts
Adding the T-shirts and sweaters, Norma found 3 + 9 = 12 clothes
Subtracting the clothes she left from the clothes she found, 27 - 12 = 15 clothes are missing
The answer is 15

Question: Adam has an orchard. Every day for 30 days he picks 4 apples from his orchard. After a month, Adam has collected all the remaining apples, which were 230. How many apples in total has Adam collected from his orchard?
Let's think step by step
During 30 days Adam picked 4 * 30 = 120 apples.
So in total with all the remaining apples, he picked 120 + 230 = 350 apples from his orchard.
The answer is 350
"""

prompt_template = {

"llama":"""<|start_header_id|>user<|end_header_id|>\n\n %s Let's think step by step.\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n""",

"qwen_base": """<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n""",

"qwen_math": """<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n""",

"qwen_math_fewshot": f"""<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{gsm8k_fewshot_prompt}%s<|im_end|>\n<|im_start|>assistant\n""",

"deepseek_math": """User: %s\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant:\n""",

"qwen_base_fewshot": f"""<|im_start|>system\nPlease reason step by step.<|im_end|>\n<|im_start|>user
{gsm8k_fewshot_prompt}\nQuestion: %s\nLet's think step by step\n<|im_end|>\n<|im_start|>assistant\n""",

"qwen3":"<|im_start|>user\n%s <|im_end|>\n<|im_start|>assistant\n",

"qwen3_fewshot":f"<|im_start|>user\n{gsm8k_fewshot_prompt}%s /think<|im_end|>\n<|im_start|>assistant\n",

"qwen3_nothink":"<|im_start|>user\n%s /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",

"qwen3_fewshot_nothink":f"<|im_start|>user\n{gsm8k_fewshot_prompt}%s /no_think<|im_end|>\n<|im_start|>assistant\n",

"mistral":"""[INST]Below is an instruction that describes a task. Write a response that appropriately completes the request.\n%s\nLet's think step by step.\n[/INST]""",

"none":"""%s"""
}

def get_prompt(tokenizer, template_index):
    
    if template_index == 'chat_template':
        prompt = "%s"
        messages = [
            {"role": "user", "content": prompt}
        ]
        template = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            thinking_budget=50,
        )
        print("template_index: chat template")
        print("----------------------- template -----------------------")
        print(template)
        print("--------------------------------------------------------")

    
    elif template_index == 'fewshot_chat_template':

        prompt = gsm8k_fewshot_prompt+"%s"
        messages = [
            {"role": "user", "content": prompt}
        ]
        template = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True 
        )
        print("template_index: chat template")
        print("----------------------- template -----------------------")
        print(template)
        print("--------------------------------------------------------")

    else:

        print("template_index:", template_index)
        print("----------------------- template -----------------------")
        print(prompt_template[template_index])
        print("--------------------------------------------------------")
        template = prompt_template[template_index]
    
    return template


def generate(inputs, model, tokenizer, model_path):

    max_length_dict = {
        "gsm8k": 1024,
        "svamp": 1024,
        "mathqa": 1024,
        "math500": 2048,
        "amc23": 2048,
        "gpqa": 2048,
        "aime": 4096,
        "boolq": 512,
        "piqa": 512,
        "piqa": 512,
        "hellaswag": 512,
    }

    for key in max_length_dict.keys():
        if key in model_path.lower():
            max_length = max_length_dict[key]
    
    if max_length is None:
        'Please set max length!'
        raise KeyError

    if "llama" in model_path.lower():
        generation_config = GenerationConfig(
            do_sample=False,
            no_repeat_ngram_size=5,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=1,
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            generation_config=generation_config
        )

    elif "qwen2.5-math-7b" in model_path.lower() or "qwen3" in model_path.lower():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
        )
    else:
        generation_config = GenerationConfig(
            do_sample=False,
            no_repeat_ngram_size=5,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=1,
        )
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            generation_config=generation_config
        )
    
    return outputs