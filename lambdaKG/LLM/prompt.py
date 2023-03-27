import inflect
inflection_engine = inflect.engine()

import spacy
nlp = spacy.load("en_core_web_sm")

def article(word):
    return "An" if word[0] in ['a', 'e', 'i', 'o', 'u'] else "A"

def vp_present_participle(phrase):
    doc = nlp(phrase)
    return ' '.join([
        inflection_engine.present_participle(token.text) if token.pos_ == "VERB" and token.tag_ != "VGG" else token.text
        for token in doc
    ])

def gen_prompt(head, relation, tail=''):
    if relation == "AtLocation":
        prompt = "You are likely to find {} {} in ".format(
            article(head), head
        )
    elif relation == "CapableOf":
        prompt = "{} can ".format(head)
    elif relation == "Causes":
        prompt = "Sometimes {} causes ".format(head)
    elif relation == "Desires":
        prompt = "{} {} desires ".format(article(head), head)
    elif relation == "HasProperty":
        prompt = "{} is ".format(head)
    elif relation == "HasSubEvent":
        prompt = "While {}, you would ".format(vp_present_participle(head))
    elif relation == "HinderedBy":
        prompt = "{}, which would not happen if ".format(head)
    elif relation == "MadeUpOf":
        prompt = "{} {} contains ".format(article(head), head)
    elif relation == "NotDesires":
        prompt = "{} {} does not desire ".format(article(head), head)
    elif relation == "ObjectUse":
        prompt = "{} {} can be used for ".format(article(head), head)
    elif relation == "isAfter":
        prompt = "{}. Before that, ".format(head)
    elif relation == "isBefore":
        prompt = "{}. After that, ".format(head)
    elif relation == "isFilledBy":
        prompt = "{}. [MASK] is filled by ".format(head) #TODO
    elif relation == "oEffect":
        prompt = "{}. The effect on others will be ".format(head)
        if 'PersonY' in head:
            prompt = prompt.replace('others', 'PersonY')
    elif relation == "oReact":
        prompt = "{}. As a result, others feel ".format(head)
        if 'PersonY' in head:
            prompt = prompt.replace('others', 'PersonY')
    elif relation == "oWant":
        if tail.strip().startswith('to'):
            prompt = "{}. After, others will want ".format(head)
        else:
            prompt = "{}. After, others will want to ".format(head)
        if 'PersonY' in head:
            prompt = prompt.replace('others', 'PersonY')
    elif relation == "xAttr":
        prompt = "{}. PersonX is ".format(head)
    elif relation == "xEffect":
        prompt = "{}. The effect on PersonX will be ".format(head)
    elif relation == "xIntent":
        if tail.strip().startswith('to'):
            prompt = "{}. PersonX did this ".format(head)
        else:
            prompt = "{}. PersonX did this to ".format(head)
    elif relation == "xNeed":
        if tail.strip().startswith('to'):
            prompt = "{}. Before, PersonX needs ".format(head)
        else:
            prompt = "{}. Before, PersonX needs to ".format(head)
    elif relation == "xReact":
        prompt = "{}. PersonX will be ".format(head)
    elif relation == "xReason":
        prompt = "{}. PersonX did this because ".format(head)
    elif relation == "xWant":
        if tail.strip().startswith('to'):
            prompt = "{}. After, PersonX will want ".format(head)
        else:
            prompt = "{}. After, PersonX will want to ".format(head)
    return prompt