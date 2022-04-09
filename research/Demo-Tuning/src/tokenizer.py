import numpy as np
import pandas as pd
from typing import List, Optional, Union


def tokenize_sent(input_text_list, tokenizer):
    sent1_token_ids = None
    sent2_token_ids = None

    text1 = input_text_list[0]
    sent1_token_ids = tokenizer.encode(text1, add_special_tokens=False)
    if len(input_text_list) == 2:
        text2 = input_text_list[1]
        text2 = text2[:1].lower() + text2[1:]
        sent2_token_ids = tokenizer.encode(' ' + text2, add_special_tokens=False)

    return sent1_token_ids, sent2_token_ids


def _tokenize_multipart_input(
    input_text_list,
    template_list,
    special_token_mapping,
    label_word_list,
    tokenizer,
    max_length,
    first_sent_limit,
    other_sent_limit,
    is_demo=False,
    demo_label_id=None,
):
    input_ids = []
    attention_mask = []
    token_type_ids = []

    segment_id = 0
    # pre-process sentence
    sent1_token_ids, sent2_token_ids = tokenize_sent(input_text_list, tokenizer=tokenizer)
    
    # truncate
    if first_sent_limit is not None:
        sent1_token_ids = sent1_token_ids[:first_sent_limit]
    if other_sent_limit is not None and sent2_token_ids is not None:
        sent2_token_ids = sent2_token_ids[:other_sent_limit]
    if max_length is not None:
        # default to truncate first sentence if have two sentence in a instance.
        # for avoiding truncate special tokens, only truncate real text.
        len_sent1 = len(sent1_token_ids)
        len_sent2 = len(sent2_token_ids) if sent2_token_ids is not None else 0
        if len_sent1 + len_sent2 > max_length:
            if len_sent2 > 0:
                sent2_token_ids = sent2_token_ids[:max_length - len_sent1]
            else:
                sent1_token_ids = sent1_token_ids[:max_length]

    for _, part in enumerate(template_list):
        new_tokens = []
        segment_plus_1_flag = False
        if part in special_token_mapping:
            new_tokens_ = []
            if is_demo and part == 'mask':
                # multi-token label
                if isinstance(label_word_list[demo_label_id], List):
                    new_tokens_.extend(label_word_list[demo_label_id])
                # single-token label
                else:
                    new_tokens_.append(label_word_list[demo_label_id])
            else:
                new_tokens_.append(special_token_mapping[part])
                
            new_tokens.extend(new_tokens_)
            if part == 'sep+':
                segment_plus_1_flag = True
        elif part[:6] == 'label_':
            label_id = int(part.split('_')[1])
            label_word = label_word_list[label_id]
            new_tokens.append(label_word)
        elif part[:5] == 'sent_':
            # Lower case the first token and discard the last token
            sent_id = int(part.split('_')[1])
            if sent_id == 0:
                new_tokens += sent1_token_ids
            else:
                new_tokens += sent2_token_ids
        else:
            # Just natural language prompt
            part = part.replace('_', ' ') 
            # handle special case when T5 tokenizer might add an extra space
            if len(part) == 1:
                new_tokens.append(tokenizer._convert_token_to_id(part))
            else:
                new_tokens += tokenizer.encode(part, add_special_tokens=False)
        
        input_ids += new_tokens
        attention_mask += [1 for i in range(len(new_tokens))]
        token_type_ids += [segment_id for i in range(len(new_tokens))]

        if segment_plus_1_flag:
            segment_id += 1
    
    return input_ids, attention_mask, token_type_ids


def tokenize_multipart_input(
    input_text_list,
    max_length,
    tokenizer,
    prompt=None,
    len_special_tokens_in_template=0,
    virtual_demo=False,
    virtual_demo_length_per_label=0,
    template=None,
    demo_template=None,
    label_word_list=None,
    first_sent_limit=None,
    other_sent_limit=None,
    demo=False,
    demo_max_length=128,
    demo_first_sent_limit=None,
    demo_other_sent_limit=None,
    num_seq_per_example=None,
):
    
    input_ids = []
    attention_mask = []
    token_type_ids = []
    mask_pos = None

    # reset `demo_max_length` and `max_length` according to `len_special_tokens_in_template`
    real_demo_max_length = demo_max_length + len_special_tokens_in_template if demo else 0
    real_max_length = max_length + len_special_tokens_in_template
    len_virtual_demo = len(label_word_list) * virtual_demo_length_per_label + 1 if virtual_demo and label_word_list is not None else 0
    total_max_length = real_max_length + len_virtual_demo + real_demo_max_length

    if prompt:
        assert template is not None

        special_token_mapping = {
            'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id, 'sep+': tokenizer.sep_token_id, 
            'prompt': tokenizer.pad_token_id
        }

        template_list = template.split('*') # Get variable list in the template

        input_ids, attention_mask, token_type_ids = _tokenize_multipart_input(
            input_text_list[:num_seq_per_example],
            max_length=max_length,
            template_list=template_list,
            special_token_mapping=special_token_mapping,
            label_word_list=label_word_list,
            tokenizer=tokenizer,
            first_sent_limit=first_sent_limit,
            other_sent_limit=other_sent_limit,
            is_demo=False,
            demo_label_id=None,
        )
        block_flag_for_demo = [0] * len(input_ids)

        # Find mask token
        mask_pos = [input_ids.index(tokenizer.mask_token_id)]
        # Make sure that the masked position is inside the max_length
        assert mask_pos[0] < real_max_length
    
        if virtual_demo and demo:
            demo_input_ids_list = []
            demo_attention_mask_list = []
            demo_token_type_ids_list = []
            demo_block_flag_for_prompt_list = []
            demo_block_flag_for_demo_list = []

            selected_indices = np.random.permutation(list(range(len(label_word_list))))[0]
            demo_label_id = selected_indices

            virtual_demo_tokens = list(range(virtual_demo_length_per_label * len(label_word_list)))
            virtual_demo_tokens.append(special_token_mapping['sep'])
            demo_block_flag_for_demo = list(range(1, virtual_demo_length_per_label * len(label_word_list) + 1))
            demo_block_flag_for_demo.append(0)

            demo_template_list = demo_template.split('*')

            demo_input_text_list = [input_text_list[selected_indices + 1]] if num_seq_per_example == 1 else input_text_list[num_seq_per_example * (selected_indices + 1): num_seq_per_example * (selected_indices + 1) + 2]
            demo_input_ids, _, _ = _tokenize_multipart_input(
                input_text_list=demo_input_text_list,
                max_length=demo_max_length,
                template_list=demo_template_list,
                special_token_mapping=special_token_mapping,
                label_word_list=label_word_list,
                tokenizer=tokenizer,
                first_sent_limit=demo_first_sent_limit,
                other_sent_limit=demo_other_sent_limit,
                is_demo=True,
                demo_label_id=demo_label_id,
            )

            demo_proto_prompt = virtual_demo_tokens[:selected_indices * virtual_demo_length_per_label] + demo_input_ids + virtual_demo_tokens[virtual_demo_length_per_label * (selected_indices + 1):]
            demo_block_flag_for_demo = demo_block_flag_for_demo[:selected_indices * virtual_demo_length_per_label] + [0]*len(demo_input_ids) + demo_block_flag_for_demo[virtual_demo_length_per_label * (selected_indices + 1):]

            demo_input_ids_list.append(demo_proto_prompt)
            demo_attention_mask_list.append([1 for i in range(len(demo_proto_prompt))])
            demo_token_type_ids_list.append([0 for i in range(len(demo_proto_prompt))])
            demo_block_flag_for_demo_list.append(demo_block_flag_for_demo)

        elif not virtual_demo and demo:
            demo_input_ids_list = []
            demo_attention_mask_list = []
            demo_token_type_ids_list = []
            demo_block_flag_for_demo_list = []

            demo_template_list = demo_template.split('*')

            for i in range(len(label_word_list)):
                demo_input_text_list = demo_input_text_list = [input_text_list[i + 1]] if num_seq_per_example == 1 else input_text_list[num_seq_per_example * (i + 1): num_seq_per_example * (i + 1) + 2]
                demo_input_ids, _, _ = _tokenize_multipart_input(
                    input_text_list=demo_input_text_list,
                    max_length=demo_max_length,
                    template_list=demo_template_list,
                    special_token_mapping=special_token_mapping,
                    label_word_list=label_word_list,
                    tokenizer=tokenizer,
                    first_sent_limit=demo_first_sent_limit,
                    other_sent_limit=demo_other_sent_limit,
                    is_demo=True,
                    demo_label_id=i,
                )

                # add [SEP]
                demo_input_ids.append(special_token_mapping['sep'])
                demo_block_flag_for_demo = [0] * (len(demo_input_ids))

                demo_input_ids_list.append(demo_input_ids)
                demo_attention_mask_list.append([1 for i in range(len(demo_input_ids))])
                demo_token_type_ids_list.append([0 for i in range(len(demo_input_ids))])
                demo_block_flag_for_demo_list.append(demo_block_flag_for_demo)

    else:
        input_ids = [tokenizer.cls_token_id]
        attention_mask = [1]
        token_type_ids = [0]
        block_flag_for_demo = [0]

        for _, input_text in enumerate(input_text_list):
            if input_text is None:
                # Do not have text_b
                continue
            if pd.isna(input_text) or input_text is None:
                # Empty input
                input_text = ''
            input_tokens = tokenizer.encode((input_text) + [tokenizer.sep_token_id], add_special_tokens=False)
            input_ids += input_tokens
            attention_mask += [1 for i in range(len(input_tokens))]
            token_type_ids += [0 for i in range(len(input_tokens))]
            block_flag_for_demo += [0 for i in range(len(input_tokens))]

    if virtual_demo and demo:
        for i in range(len(demo_input_ids_list)):
            demo_input_ids_list[i] = input_ids + demo_input_ids_list[i]
            demo_input_ids_list[i] += [tokenizer.pad_token_id] * (total_max_length - len(demo_input_ids_list[i]))
            demo_attention_mask_list[i] = attention_mask + demo_attention_mask_list[i]
            demo_attention_mask_list[i] += [0] * (total_max_length - len(demo_attention_mask_list[i]))
            demo_token_type_ids_list[i] = token_type_ids + demo_token_type_ids_list[i]
            demo_token_type_ids_list[i] += [0] * (total_max_length - len(demo_token_type_ids_list[i]))
            demo_block_flag_for_demo_list[i] = block_flag_for_demo + demo_block_flag_for_demo_list[i]
            demo_block_flag_for_demo_list[i] += [0] * (total_max_length - len(demo_block_flag_for_demo_list[i]))

        virtual_demo_tokens = list(range(virtual_demo_length_per_label * len(label_word_list)))
        virtual_demo_tokens.append(special_token_mapping['sep'])
        block_flag_for_demo_ = list(range(1, virtual_demo_length_per_label * len(label_word_list) + 1))
        block_flag_for_demo_.append(0)

        input_ids += virtual_demo_tokens
        attention_mask += [1 for i in range(len(virtual_demo_tokens))]
        token_type_ids += [0 for i in range(len(virtual_demo_tokens))]
        block_flag_for_demo += block_flag_for_demo_

    elif not virtual_demo and demo:
        for i in range(len(demo_input_ids_list)):
            input_ids += demo_input_ids_list[i]
            attention_mask += demo_attention_mask_list[i]
            token_type_ids += demo_token_type_ids_list[i]
            block_flag_for_demo += demo_block_flag_for_demo_list[i]

    input_ids += [tokenizer.pad_token_id] * (total_max_length - len(input_ids))
    attention_mask += [0] * (total_max_length - len(attention_mask))
    token_type_ids += [0] * (total_max_length - len(token_type_ids))
    block_flag_for_demo += [0] * (total_max_length - len(block_flag_for_demo))

    assert (len(input_ids) == len(attention_mask)) & (len(input_ids) == len(block_flag_for_demo))

    result = {'input_ids': input_ids, 'attention_mask': attention_mask, 'block_flag_for_demo': block_flag_for_demo}
    if 'BERT' in type(tokenizer).__name__:
        # Only provide token type ids for BERT
        result['token_type_ids'] = token_type_ids

    if prompt:
        result['mask_pos'] = mask_pos

    demo_result = None
    if virtual_demo and demo:
        demo_result = []
        for i in range(len(demo_input_ids_list)):
            result_ = {'input_ids': demo_input_ids_list[i], 'attention_mask': demo_attention_mask_list[i], 'block_flag_for_demo': demo_block_flag_for_demo_list[i]}
            if prompt:
                result_['mask_pos'] = mask_pos
            
            if 'BERT' in type(tokenizer).__name__:
                # Only provide token type ids for BERT
                result_['token_type_ids'] = token_type_ids

            demo_result.append(result_)
    # print(result)
    return result, demo_result
