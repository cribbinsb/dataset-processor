try:
    import ollama
    ollama_ok=True
except ImportError:
    ollama_ok=False

def get_description(jpg):

    jpg_file="/tmp/tmp.jpg"
    with open(jpg_file, 'wb') as f:
        f.write(jpg)

    prompt_str=""
    prompt_str+="Give a description of person in center of image. Include-is the image clear or blurry."
    prompt_str+="Include a list of attributes: sex, age, main overall top half color(s), overall bottom half color(s)."
    prompt_str+="Yes/no for if a bag is carried, a hat/head covering worn, beard visible, face covering worn,"
    prompt_str+="if long hair, if glasses worn, if coat/jacket worn."
    prompt_str+="Add no other details. Do not make things up, only put what is true."
    message={'role':'user',
            'content': prompt_str,
            'images': [jpg_file]}
    response = ollama.chat(
        #model='llama3.2-vision',
        model='minicpm-v',
        messages=[message]
        )
    desc=response['message']['content']
    return desc

def direct_attributes(jpg, attrs_main, attrs_colour): 
    jpg_file="/tmp/tmp.jpg"
    with open(jpg_file, 'wb') as f:
        f.write(jpg)

    prompt_str =""
    prompt_str += "Consider the image of a person and fill in binary attributes accurately."
    prompt_str += "Return answer strictly in JSON format, for each key that applies "
    prompt_str += "to the image return that key with value true. For attributes that are "
    prompt_str += "not true for the person pictured, or where it is unclear you must "
    prompt_str += "return false."
    prompt_str += "Attribute list to be returned as JSON keys:\n"
    message={'role':'user',
            'content': prompt_str,
            'images': [jpg_file]}
    response = ollama.chat(
        #model='llama3.2-vision',
        model='minicpm-v',
        messages=[message]
        )
    desc=response['message']['content']
    return desc

def get_attributes_from_description(desc, attrs_main, attrs_colour):         
    prompt_str =""
    prompt_str += "Given the following description, fill in binary attributes accurately."
    prompt_str += "Return answer strictly in JSON format, for each key that applies "
    prompt_str += "to the image return that key with value true. "
    prompt_str += " Do not make up information. For example if there "
    prompt_str += "is no mention or relevant information the key must not be included.\n"
    prompt_str += "Attribute list to be returned as JSON keys:\n"
    for a in attrs_main:
        prompt_str+= a+" ,"

    prompt_str += "\nDescription:\n"
    prompt_str += desc

    message={'role':'user',
            'content': prompt_str,
            }
    response = ollama.chat(
        model='llama3.2',
        #model='minicpm-v',
        messages=[message]
    )

    attr_r1=response['message']['content']

    prompt_str =""
    prompt_str += "Given the following description, we are interested in the colours"
    prompt_str += "mainly present in the person's halves: top/shirt and bottom/pants."
    prompt_str += "Return answer srictly in JSON format with no extra output with ALL of "
    prompt_str += "the following boolean keys."
    prompt_str += "Choose only from true or false for each key. Put false if the attribute "
    prompt_str += " is not present OR cannot"
    prompt_str += " be accurately determined. Do not make up information.\n"
    prompt_str += "JSON keys:\n"
                             
    for a in attrs_colour:
        prompt_str+=a+" ,"

    prompt_str += "\nDescription:\n"
    prompt_str += desc

    message={'role':'user',
            'content': prompt_str,
            }
    response = ollama.chat(
        model='llama3.2',
        #model='minicpm-v',
        messages=[message]
    )
    attr_r2=response['message']['content']

    attr_r=attr_r1+"\n"+attr_r2

    return attr_r

class LLMOllama:
    def __init__(self):
        assert ollama_ok, "Try pip install ollama"
        pass

    def get_batch(self):
        return 8

    def generate_attributes(self,attrs, jpegs):
        attrs_main=[]
        attrs_colour=[]
        for a in attrs:
            a=a.split(":")[1]
            if "top" in a or "bottom" in a:
                attrs_colour.append(a)
            else:
                attrs_main.append(a)
        out=[]
        if False:
            for j in jpegs:
                r=direct_attributes(j, attrs_main, attrs_colour)
                out.append(r)
        else:
            desc=[]
            for j in jpegs:
                desc.append(get_description(j))
            out=[]
            for d in desc:
                r=get_attributes_from_description(d, attrs_main, attrs_colour)
                out.append(r)

        for i,r in enumerate(out):
            #hacky fixups for common LLM mistakes
            r=r.replace("\n", ",")
            r=r.replace("{","")
            r=r.replace("}","")
            r=r.replace("\n",",")
            r=r.replace(",,",",")
            r=r.replace(",,",",")
            r=r.replace("null","false")
            r=r.replace("unknown","false")
            r=r.replace("not_applicable","false")
            r=r.replace("white", "light")
            r=r.replace("pink", "red")
            out[i]=r
        return out
