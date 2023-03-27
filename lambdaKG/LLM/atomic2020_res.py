import openai
import jsonlines
import time
from LLM.prompt import gen_prompt
from easyinstruct import BasePrompt
from easyinstruct.utils import set_openai_key, set_proxy

instruct_prompt=f"Commonsense reasoning:\n{}\n{}"
api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

set_openai_key(api_key)
set_proxy("http://127.0.0.1:7890")

gpt_args = {
    "engine": "text-davinci-001",
    "temperature": 0,
    "max_tokens": 32,
    "stop": ['.','\n'],
    "frequency_penalty": 1.0,
}

def get_prompt(line):
    query = line['query']
    examples = line['examples']
    query_ = "Then: "+gen_prompt(*query.split(' @@ '))
    new_exs = []
    for ex in examples:
        new_ex = ex.strip().strip('.').split('\t')
        if len(new_ex) == 2:
            new_ex.append('none')
        if new_ex[2].strip().startswith('to'):
            dd = new_ex[2].split('to',1)[-1].strip()
        else:
            dd = new_ex[2].strip()
        new_ex = "Example: "+gen_prompt(*new_ex[:2])+ new_ex[2] +'. \n'
        new_exs.append(new_ex)
    examples_ = ''.join(new_exs)
    prompt = instruct_prompt.format(examples_, query_)
    if "[MASK]" in prompt:
        prompt = prompt.strip().replace("___", "[MASK]")
    else:
        prompt = prompt.strip().replace("___", "something or someone")
    return prompt

def get_output(prompt, line):
    try:
        prompts = BasePrompt()
        prompts.build_prompt(prompt)
        response = prompts.get_openai_result(**gpt_args)
        time.sleep(1.5)
    except:
        print("An error occurred while calling the OpenAI API for text-davinci-003!")
        
    res = response["choices"][0]["text"]
    line['result'] = res
    return line

    
if __name__ == '__main__':
    data = []
    with jsonlines.open('dataset/atomic_2020_data/test.jsonl') as fp:
        for line in fp:
            data.append(line)
    all_res = []
    for line in data:
        prompt = get_prompt(line)
        line = get_output(prompt, line)
        all_res.append(line)
        
#     with jsonlines.open('test_result.jsonl', 'a') as fp:
#         for line in all_res:
#             fp.write(line)

    results = dict()
    for line in all_res:
        results[line['query']] = line

    results_process = []
    for line in results.keys():
        res = results[line]
        results_process.append(dict(
            head = res['query'].split(' @@ ')[0],
            relation = res['query'].split(' @@ ')[1],
            tails = res['answer'],
            generations = [res['result'].strip()],
            greedy = res['result'].strip(),
        ))

    with jsonlines.open('dataset/atomic_2020_data/test_result.json', 'w') as fp:
        fp.write_all(results_process)